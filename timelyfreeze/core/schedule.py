import time
from matplotlib import pyplot as plt
import numpy as np
import torch
import torch.distributed as dist
from typing import List, Dict, Tuple
from scipy.optimize import linprog
from torchtitan.tools.logging import logger

from .action import Action, ActionType, ActionWithTime, ActionWithLog, ActionWithFreezing
from .util import draw_pipeline_schedule, get_abs_path
from .config import TimelyFreezeConfig, Comm
from scipy.optimize import minimize, LinearConstraint, Bounds


def gather_pipeline_schedule(log_schedule:List[ActionWithLog], comm: Comm, log_window :int|None = None)-> List[List[ActionWithTime]]:
    ''' 
    Gather the pipeline schedule from all ranks and create a pipeline schedule with freezing actions.
    If the rank is not the last stage, it will return an empty list.
    '''
    # requires communication among all ranks
    # dist.barrier(group=args.pp_group, device_ids=[args.local_rank])

    # collect the number of actions per rank
    device = f'cuda:{comm.local_rank}'
    num_actions = len(log_schedule)
    local_num_actions = torch.tensor(num_actions, device=device, dtype=torch.int32)
    num_actions_all_tensor = torch.empty(comm.pp, device=device, dtype=torch.int32)
    dist.all_gather_into_tensor(num_actions_all_tensor, local_num_actions, group=comm.pp_group)
    num_actions_all = num_actions_all_tensor.cpu().tolist()  # [num_pipelines]

    # determine schedule info width robustly across ranks
    schedule_info_local = int(len(log_schedule[0].to_tensor(log_window=log_window))) if num_actions > 0 else 0
    schedule_info_local_t = torch.tensor(schedule_info_local, device=device, dtype=torch.int32)
    schedule_info_all_t = torch.empty(comm.pp, device=device, dtype=torch.int32)
    dist.all_gather_into_tensor(schedule_info_all_t, schedule_info_local_t, group=comm.pp_group)
    schedule_info = int(schedule_info_all_t.max().item())

    # create a tensor to send/gather from all ranks: (schedule info) x (max num_actions)
    max_actions = int(max(num_actions_all)) if len(num_actions_all) > 0 else 0
    dtype = torch.float16  # ActionWithLog.to_tensor returns float16
    data_to_gather = torch.zeros((schedule_info, max_actions), device=device, dtype=dtype)
    if num_actions > 0 and schedule_info > 0:
        actions_tensor = torch.stack([action.to_tensor(log_window=log_window) for action in log_schedule]).to(device)
        data_to_gather[:, :num_actions] = actions_tensor.T

    gathered = torch.empty((comm.pp, schedule_info, max_actions), device=device, dtype=dtype)
    dist.all_gather_into_tensor(gathered, data_to_gather, group=comm.pp_group)
    data_gather_list = gathered.cpu()  # [num_pipelines, schedule, num_actions]

    # create a pipeline schedule in class ActionWithTime
    pipeline_schedule: List[List[ActionWithTime]] = [
        [ActionWithTime(*action.tolist()) for action in rank_list.T[:num_actions_all[r]]]
        for r, rank_list in enumerate(data_gather_list)
    ]
    return link_actions(pipeline_schedule)
    
def link_actions(pipeline_schedule:List[List[ActionWithTime]])-> List[List[ActionWithTime]]:
    '''Edge Construction
    Link the actions in the pipeline schedule based on the microbatch, stage, and action type.
    This function will set the prev_actions and next_actions attributes of each action.
    '''
    
    if any(action.prev_actions for actions_per_rank in pipeline_schedule for action in actions_per_rank) \
        or any(action.next_actions for actions_per_rank in pipeline_schedule for action in actions_per_rank):
            return pipeline_schedule # already linked, return the original schedule
        
    action_dict: Dict[tuple, ActionWithTime] = {(action.type, action.rank, action.microbatch, action.stage): action \
                                                    for actions_per_rank in pipeline_schedule for action in actions_per_rank}
    def set_links(prev, nxt):
        '''set the prev/next links between actions'''
        if prev is None or nxt is None:
            logger.debug(f"prev or nxt is None. prev: {prev}, nxt: {nxt}")
            return
        if nxt in prev.next_actions:
            logger.debug(f"nxt is already in prev.next_actions. nxt: {nxt}, prev: {prev}")
            return # already linked, do nothing
        prev.next_actions.append(nxt)
        nxt.prev_actions.append(prev)
        return 
    
    num_ranks: int = len(pipeline_schedule) # same as args.pp
    # compute last_stage and num_microbatches in a single pass, robust to varying ranks
    last_stage: int = -1
    max_microbatch: int = -1
    for actions_per_rank in pipeline_schedule:
        for action in actions_per_rank:
            if action.microbatch is not None and action.microbatch > max_microbatch:
                max_microbatch = action.microbatch
            if action.type == ActionType.FORWARD and action.stage is not None and action.stage > last_stage:
                last_stage = action.stage
    if last_stage < 0:
        last_stage = 0
    num_microbatches: int = max_microbatch + 1

    # add prev/next links between actions
    get_action = action_dict.get
    for rank_actions in pipeline_schedule:
        for itr, action in enumerate(rank_actions):
            # next scheduled action
            if itr < len(rank_actions)-1: # if not the last action in the rank
                next_action = rank_actions[itr+1]
                set_links(action, next_action)
                
            # microbatch conditions
            if action.microbatch < num_microbatches-1: # microbatch n should be executed after microbatch n-1
                next_action = get_action((action.type, action.rank, action.microbatch+1, action.stage))
                set_links(action, next_action)
            
            # action type conditions
            if action.type == ActionType.FORWARD: # full-backward / backward-input should be executed after forward
                next_action = get_action((ActionType.FULL_BACKWARD, action.rank, action.microbatch, action.stage))
                if next_action is None:
                    next_action = get_action((ActionType.BACKWARD_INPUT, action.rank, action.microbatch, action.stage))
                set_links(action, next_action)
            elif action.type == ActionType.BACKWARD_INPUT: # backward-weight should be executed after backward-input
                next_action = get_action((ActionType.BACKWARD_WEIGHT, action.rank, action.microbatch, action.stage))
                set_links(action, next_action)
                
            # stage conditions
            if action.type == ActionType.FORWARD and action.stage < last_stage: # forward of stage n should be executed after stage n-1
                next_action = get_action((action.type, (action.rank+1)%num_ranks, action.microbatch, action.stage+1))
                if next_action is None:
                    next_action = get_action((action.type, (action.rank-1+num_ranks)%num_ranks, action.microbatch, action.stage+1))
                set_links(action, next_action)
            if action.type in [ActionType.FULL_BACKWARD, ActionType.BACKWARD_INPUT] and action.stage > 0: # full-backward / backward-input of stage n should be executed after stage n+1
                next_action = get_action((ActionType.FULL_BACKWARD, (action.rank-1+num_ranks)%num_ranks, action.microbatch, action.stage-1))
                if next_action is None:
                    next_action = get_action((ActionType.FULL_BACKWARD, (action.rank+1)%num_ranks, action.microbatch, action.stage-1))
                if next_action is None:
                    next_action = get_action((ActionType.BACKWARD_INPUT, (action.rank-1+num_ranks)%num_ranks, action.microbatch, action.stage-1))
                if next_action is None:
                    next_action = get_action((ActionType.BACKWARD_INPUT, (action.rank+1)%num_ranks, action.microbatch, action.stage-1))
                set_links(action, next_action)
    return pipeline_schedule


def solve_dag_lp(pipeline_schedule:List[List[ActionWithFreezing]], max_freeze_ratio: float=0.9)-> List[List[ActionWithFreezing]]:
    '''
    Updated on July 29, 2025.
    Solve the DAG LP problem to find the optimal schedule for the pipeline schedule.
    This function assumes that each action in the pipeline schedule is all linked as a DAG and has max_duration and min_duration.
    This function returns the pipeline schedule with the start time and end time of each action set.
    '''
    # Build problem without NetworkX using index mapping
    start_node = ActionWithFreezing(ActionType.START, 0, 0, 0, 0)
    end_node = ActionWithFreezing(ActionType.FINISH, 0, 0, 0, 0)
    actions_flat = [action for actions_per_rank in pipeline_schedule for action in actions_per_rank]
    m = len(actions_flat)
    actions_all = [start_node] + actions_flat + [end_node]
    n = len(actions_all)
    num_stages = max(action.stage for action in actions_flat) + 1 if m > 0 else 1
    assert n == 2 + sum(len(rank_actions) for rank_actions in pipeline_schedule), f"num_actions: {n}, sum of rank actions: {sum(len(rank_actions) for rank_actions in pipeline_schedule)}"

    def action_key(a: ActionWithFreezing):
        return (a.type, a.rank, a.microbatch, a.stage)
    key_to_idx = {action_key(a): i+1 for i, a in enumerate(actions_flat)}

    # edges (u->v) with indices in [0, n-1]
    edges = []
    for i, a in enumerate(actions_flat):
        u = i + 1
        if len(a.prev_actions) == 0:
            edges.append((0, u))
        if len(a.next_actions) == 0:
            edges.append((u, n - 1))
        for na in a.next_actions:
            v = key_to_idx.get(action_key(na))
            if v is not None:
                edges.append((u, v))

    num_vars = 2 * n
    c = np.zeros(num_vars, dtype=float)
    c[n - 1] = 999999.0

    w_min = np.zeros(n, dtype=float)
    w_max = np.zeros(n, dtype=float)
    freezable = np.zeros(n, dtype=bool)
    stages = np.full(n, -1, dtype=int)
    for i, a in enumerate(actions_flat, start=1):
        w_min[i] = a.min_duration
        w_max[i] = a.max_duration
        freezable[i] = a.freezable and (a.min_duration < a.max_duration)
        stages[i] = a.stage
    valid = freezable & (w_max > w_min)
    denom = np.where(valid, (w_max - w_min), 1.0)
    c[n:n + n][valid] = -1.0 / denom[valid]

    A, b = [], []
    for (u, v) in edges:
        row = np.zeros(num_vars, dtype=float)
        row[u] = 1.0
        row[v] = -1.0
        row[n + u] = 1.0
        A.append(row)
        b.append(0.0)

    row0 = np.zeros(num_vars, dtype=float)
    row0[0] = 1.0
    row0[n] = 1.0
    A.append(row0)
    b.append(0.0)

    for s in range(num_stages):
        idxs = np.where((stages == s) & valid)[0]
        if idxs.size == 0:
            continue
        row = np.zeros(num_vars, dtype=float)
        row[n + idxs] = -1.0 / (w_max[idxs] - w_min[idxs])
        A.append(row)
        sum_term = float(np.sum(w_max[idxs] / (w_max[idxs] - w_min[idxs])))
        b.append(max_freeze_ratio * idxs.size - sum_term)

    bounds = [(0, None)] * n
    bounds += [(float(w_min[i]), float(w_max[i])) for i in range(n)]

    res = linprog(c, A_ub=np.array(A, dtype=float), b_ub=np.array(b, dtype=float), bounds=bounds, method="highs")

    if res.success:
        durations = res.x[n:2 * n]
        for i, a in enumerate(actions_flat, start=1):
            a.duration = float(durations[i])
    else:
        raise RuntimeError("Linear programming failed.")
    return schedule_pipeline(pipeline_schedule)


def solve_dag_qp(pipeline_schedule:List[List[ActionWithFreezing]], max_freeze_ratio: float=0.9)-> List[List[ActionWithFreezing]]:
        '''
        Solve the DAG QP problem (was LP). This variant uses a quadratic program
        to obtain a smoother/regularized solution for action durations while
        keeping the same linear constraints as the original LP formulation.

        Variables:
          x[0:n]   -> start times t_i for each node (including start/end)
          x[n:2n]  -> durations w_i for each node

        Objective (QP):
          minimize  big_w * t_{n-1}  - sum_i alpha_i * w_i  + 0.5 * gamma * sum_i (w_i^2)
        where alpha_i = 1 / (w_max[i] - w_min[i]) for freezable valid actions (as before),
        gamma is a small regularization parameter to make the problem strictly convex.

        Constraints:
          For each edge (u->v): t_u + w_u <= t_v
          Start-node constraint: t_0 + w_0 <= 0
          Per-stage freeze ratio linear constraints preserved from LP:
             sum_i ( - w_i / (w_max[i] - w_min[i]) ) <= max_freeze_ratio * count - sum(w_max / (w_max - w_min))

        Bounds:
          t_i >= 0
          w_min[i] <= w_i <= w_max[i]
        '''

        # Build problem indices and edges (same mapping as previous LP)
        start_node = ActionWithFreezing(ActionType.START, 0, 0, 0, 0)
        end_node = ActionWithFreezing(ActionType.FINISH, 0, 0, 0, 0)
        actions_flat = [action for actions_per_rank in pipeline_schedule for action in actions_per_rank]
        m = len(actions_flat)
        actions_all = [start_node] + actions_flat + [end_node]
        n = len(actions_all)
        num_stages = max((action.stage for action in actions_flat), default=0) + 1 if m > 0 else 1
        assert n == 2 + sum(len(rank_actions) for rank_actions in pipeline_schedule), f"num_actions: {n}, sum of rank actions: {sum(len(rank_actions) for rank_actions in pipeline_schedule)}"

        def action_key(a: ActionWithFreezing):
            return (a.type, a.rank, a.microbatch, a.stage)
        key_to_idx = {action_key(a): i+1 for i, a in enumerate(actions_flat)}

        edges = []
        for i, a in enumerate(actions_flat):
            u = i + 1
            if len(a.prev_actions) == 0:
                edges.append((0, u))
            if len(a.next_actions) == 0:
                edges.append((u, n - 1))
            for na in a.next_actions:
                v = key_to_idx.get(action_key(na))
                if v is not None:
                    edges.append((u, v))

        # Prepare parameters used in objective and bounds
        w_min = np.zeros(n, dtype=float)
        w_max = np.zeros(n, dtype=float)
        freezable = np.zeros(n, dtype=bool)
        stages = np.full(n, -1, dtype=int)
        for i, a in enumerate(actions_flat, start=1):
            w_min[i] = a.min_duration
            w_max[i] = a.max_duration
            freezable[i] = a.freezable and (a.min_duration < a.max_duration)
            stages[i] = a.stage
        valid = freezable & (w_max > w_min)
        denom = np.where(valid, (w_max - w_min), 1.0)

        # alpha weights (same sign as LP): LP used -1/denom in c to encourage freezing,
        # so we keep -alpha * w in objective (alpha = 1/denom).
        alpha = np.zeros(n, dtype=float)
        alpha[valid] = 1.0 / denom[valid]

        # objective parameters
        big_w = 1e6  # large weight on makespan (t_{n-1})
        gamma = 1e-3  # small quadratic regularization on durations to obtain QP

        # Build linear constraints A_ub x <= b_ub similar to LP; we'll convert to LinearConstraint
        num_vars = 2 * n
        A_ub = []
        b_ub = []
        for (u, v) in edges:
            row = np.zeros(num_vars, dtype=float)
            row[u] = 1.0        # t_u
            row[v] = -1.0       # -t_v
            row[n + u] = 1.0    # +w_u
            A_ub.append(row)
            b_ub.append(0.0)

        # start node constraint
        row0 = np.zeros(num_vars, dtype=float)
        row0[0] = 1.0
        row0[n + 0] = 1.0
        A_ub.append(row0)
        b_ub.append(0.0)

        # per-stage freeze ratio constraints (preserve LP formulation)
        for s in range(num_stages):
            idxs = np.where((stages == s) & valid)[0]
            if idxs.size == 0:
                continue
            row = np.zeros(num_vars, dtype=float)
            # for each duration variable index j in idxs, set coefficient for w_j
            row[n + idxs] = -1.0 / (w_max[idxs] - w_min[idxs])
            A_ub.append(row)
            sum_term = float(np.sum(w_max[idxs] / (w_max[idxs] - w_min[idxs])))
            b_ub.append(max_freeze_ratio * idxs.size - sum_term)

        A_ub = np.array(A_ub, dtype=float)
        b_ub = np.array(b_ub, dtype=float)

        # Bounds: t_i >= 0 ; w_i in [w_min, w_max]
        lb = np.zeros(num_vars, dtype=float)
        ub = np.full(num_vars, np.inf, dtype=float)
        # durations bounds
        for i in range(n):
            lb[n + i] = float(w_min[i])
            ub[n + i] = float(w_max[i])

        bounds = Bounds(lb, ub)

        # Initial guess: start times zeros, durations = w_max (safe feasible)
        x0 = np.zeros(num_vars, dtype=float)
        x0[n:2*n] = w_max.copy()

        # Ensure initial guess satisfies linear inequalities A_ub x0 <= b_ub.
        # If not, try shrink durations to w_min gradually until feasible.
        if A_ub.size > 0:
            violated = (A_ub @ x0) - b_ub
            if np.any(violated > 1e-8):
                # simple projection on durations: linearly reduce durations proportionally
                # until all constraints satisfied or reach w_min.
                max_steps = 100
                for _ in range(max_steps):
                    violated = (A_ub @ x0) - b_ub
                    if not np.any(violated > 1e-8):
                        break
                    shrink_mask = (w_max > w_min)
                    if not np.any(shrink_mask):
                        break
                    x0[n:2*n][shrink_mask] = np.maximum(w_min[shrink_mask], 0.9 * x0[n:2*n][shrink_mask] + 0.1 * w_min[shrink_mask])
        # Define objective and gradient
        def obj(x):
            t_end = x[n - 1]
            w = x[n:2*n]
            linear_term = -np.dot(alpha, w)
            quad_term = 0.5 * gamma * np.dot(w, w)
            return big_w * t_end + linear_term + quad_term

        def jac(x):
            grad = np.zeros_like(x, dtype=float)
            # derivative wrt t_end
            grad[n - 1] = big_w
            # derivative wrt durations w
            w = x[n:2*n]
            grad[n:2*n] = -alpha + gamma * w
            return grad

        # Linear constraints: A_ub x <= b_ub  --> transform to LinearConstraint with upper=b_ub and lower=-inf
        if A_ub.size > 0:
            lin_con = LinearConstraint(A_ub, -np.inf * np.ones_like(b_ub), b_ub)
            constraints = [lin_con]
        else:
            constraints = []

        # Solve with trust-constr which handles bounds and linear constraints well
        res = minimize(fun=obj, x0=x0, method='trust-constr', jac=jac, constraints=constraints, bounds=bounds,
                       options={'maxiter': 1000, 'gtol': 1e-6, 'verbose': 0})

        if res.success:
            x_opt = res.x
            durations = x_opt[n:2*n]
            for i, a in enumerate(actions_flat, start=1):
                a.duration = float(durations[i])
        else:
            raise RuntimeError(f"Quadratic programming failed: {res.message}")

        return schedule_pipeline(pipeline_schedule)


def schedule_pipeline(pipeline_schedule:List[List[ActionWithTime]],
        fwd_time:List[float]=None, bwd_time:List[float]=None, bwd_input_time:List[float]=None, bwd_weight_time:List[float]=None
    )-> List[List[ActionWithTime]]:
    '''
    Set the start time and end time of each action in the pipeline schedule.
    Update the duration of each action if fwd_time, bwd_time, bwd_input_time, bwd_weight_time are provided.
    '''
    num_ranks = len(pipeline_schedule) # same as args.pp
    num_stages = max([action.stage for actions_per_rank in pipeline_schedule for action in actions_per_rank if action.stage is not None]) + 1 # +1 for the last stage
    assert fwd_time is None or len(fwd_time) == num_stages, f"fwd_time should have the same length as num_stages. fwd_time: {len(fwd_time)}, num_stages: {num_stages}"
    assert bwd_time is None or len(bwd_time) == num_stages, f"bwd_time should have the same length as num_stages. bwd_time: {len(bwd_time)}, num_stages: {num_stages}"
    assert bwd_input_time is None or len(bwd_input_time) == num_stages, f"bwd_input_time should have the same length as num_stages. bwd_input_time: {len(bwd_input_time)}, num_stages: {num_stages}"
    assert bwd_weight_time is None or len(bwd_weight_time) == num_stages, f"bwd_weight_time should have the same length as num_stages. bwd_weight_time: {len(bwd_weight_time)}, num_stages: {num_stages}"

    # initialize the start_time, scheduled_flag and duration of each action
    for actions_per_rank in pipeline_schedule:
        for action in actions_per_rank:
            action.start_time = 0.0 # start time is 0 for all actions
            action.scheduled_flag = False # initialize the scheduled_flag to False
            if action.type == ActionType.FORWARD and fwd_time is not None:
                action.duration = fwd_time[action.stage] # set the duration based on fwd_time
            elif action.type == ActionType.FULL_BACKWARD and bwd_time is not None:
                action.duration = bwd_time[action.stage]
            elif action.type == ActionType.BACKWARD_INPUT and bwd_input_time is not None:
                action.duration = bwd_input_time[action.stage]
            elif action.type == ActionType.BACKWARD_WEIGHT and bwd_weight_time is not None:
                action.duration = bwd_weight_time[action.stage]
    
    pipeline_schedule = link_actions(pipeline_schedule) # link the actions again after setting the duration
    
    def ready_schedule(action:Action)->bool:  
        '''Decide whether given action is ready to be scheduled or not.'''
        if len(action.prev_actions)==0 or all([prev.scheduled_flag for prev in action.prev_actions]):
            return True
        return False

    def set_following_schedule_time(action:Action):
        '''Decide the start time of following schedules of the given action.'''
        assert action.scheduled_flag, f"action.start_time should have been assigned. action: {action}"
        for next_action in action.next_actions:
            next_action.start_time = max(next_action.start_time, action.end_time)
            if next_action.rank != action.rank:
                next_action.scheduled_flag = False # need to be rescheduled
        return
    
    rank_itr = [0] * num_ranks
    last_time = [0] * num_ranks
    ready_ranks = [rank for rank in range(num_ranks)]
    done_ranks = set()
    # schedule the start/end time of every action in the pipeline schedule 
    while len(done_ranks) < num_ranks:
        assert len(ready_ranks) > 0, f"ready_ranks should not be empty. ready_ranks: {ready_ranks}, done_ranks: {done_ranks}, rank_itr: {rank_itr}"
        rank = min(ready_ranks, key=lambda r: last_time[r])
        itr = rank_itr[rank]
        curr_action = pipeline_schedule[rank][itr]

        # schedule the action and delay the following actions
        curr_action.start_time = max(curr_action.start_time, last_time[rank])
        curr_action.schedule()
        set_following_schedule_time(curr_action)
        
        # find the first action readied to be scheduled in each rank
        rank_itr = [None] * num_ranks
        for rank, actions_per_rank in enumerate(pipeline_schedule):
            for itr, action in enumerate(actions_per_rank):
                if not action.scheduled_flag: 
                    rank_itr[rank] = itr
                    last_time[rank] = actions_per_rank[itr-1].end_time if itr > 0 else 0
                    break
        done_ranks = set([rank for rank in range(num_ranks) if rank_itr[rank] is None])
        ready_ranks = [rank for rank in range(num_ranks) if not rank in done_ranks and ready_schedule(pipeline_schedule[rank][rank_itr[rank]])]
    return pipeline_schedule


def set_freeze_ratio(pipeline_schedule:List[List[ActionWithTime]], config: TimelyFreezeConfig)-> List[List[ActionWithFreezing]]:
    '''
    Updated on July 29, 2025.
    Assuming the given pipeline_schedule is all scheduled. e.g., Start time and duration of each action are all set.
    Control the freeze ratio of each action in the pipeline schedule to minimize the batch time.
    '''

    # cast the pipeline_schedule to ActionWithFreezing
    pipeline_schedule_freezing: List[List[ActionWithFreezing]] = []
    if isinstance(pipeline_schedule[0][0], ActionWithFreezing):
        pipeline_schedule_freezing = pipeline_schedule # already casted to ActionWithFreezing
    else:
        pipeline_schedule_freezing = [[ActionWithFreezing(action.type, action.rank, action.microbatch, action.stage, action.duration) for action in rank_actions] for rank_actions in pipeline_schedule]
    
    def set_expected_freeze_ratio(pipeline_schedule:List[List[ActionWithTime]], ratio:float=0.0, last_block_ratio:float=0.9)-> List[List[ActionWithTime]]:
        '''
        Initialize the duration of each action in the pipeline schedule by ratio.
        The ratio is used to set the duration of each action based on its min and max duration.
        '''
        for rank_actions in pipeline_schedule:
            for itr, action in enumerate(rank_actions):
                if action.type in [ActionType.FULL_BACKWARD, ActionType.BACKWARD_WEIGHT]:
                    if itr == len(rank_actions)-1: # if the last action in the rank, set the efr to 0.9
                        action.expected_freeze_ratio = max(ratio, last_block_ratio)
                    else:
                        action.expected_freeze_ratio = ratio
        return pipeline_schedule

    # initialize the duration of each action to have a maximum batch time
    timestamp = time.strftime("%y%m%d_%H%M")
    pipeline_schedule_freezing = schedule_pipeline(set_expected_freeze_ratio(pipeline_schedule_freezing, ratio=0, last_block_ratio=0))
    max_batch_time = max([rank_actions[-1].end_time for rank_actions in pipeline_schedule_freezing]) # get the maximum batch time
    if config.comm.is_last_stage and config.metrics.draw_graph:
        draw_pipeline_schedule(save_file=f'pipeline_schedule/{timestamp}_max_batch_time.svg',
                            config=config,
                            pipeline_schedule=pipeline_schedule_freezing,
                            title=f"Max Batch Time: {max_batch_time:.2f} ms",
                            xlabel="Time (ms)", ylabel="Rank", tick_unit=200
                            )
        
    # initialize the duration of each action to have a minimum batch time
    pipeline_schedule_freezing = schedule_pipeline(set_expected_freeze_ratio(pipeline_schedule_freezing, ratio=1))
    min_batch_time = max([rank_actions[-1].end_time for rank_actions in pipeline_schedule_freezing]) # get the minimum batch time
    if config.comm.is_last_stage and config.metrics.draw_graph:
        draw_pipeline_schedule(save_file=f'pipeline_schedule/{timestamp}_min_batch_time.svg',
                            config=config,
                            pipeline_schedule=pipeline_schedule_freezing,
                            title=f"Min Batch Time: {min_batch_time:.2f} ms",
                            xlabel="Time (ms)", ylabel="Rank", tick_unit=200
                            )

    # calculate the expected freeze ratio based on the maximum and minimum batch time and schedule the pipeline.
    max_freeze_ratio = getattr(config.freezing, "max_freeze_ratio", 0.9)
    pipeline_schedule_freezing = solve_dag_qp(pipeline_schedule_freezing, max_freeze_ratio=max_freeze_ratio) # solve the DAG LP problem to find the optimal schedule

    # if config.comm.is_last_stage:
    batch_time = max([rank_actions[-1].end_time for rank_actions in pipeline_schedule_freezing])
    average_freeze_ratio = sum([action.expected_freeze_ratio for rank_actions in pipeline_schedule_freezing for action in rank_actions if action.type in [ActionType.BACKWARD_WEIGHT, ActionType.FULL_BACKWARD]]) / sum([1 for rank_actions in pipeline_schedule_freezing for action in rank_actions if action.type in [ActionType.BACKWARD_WEIGHT, ActionType.FULL_BACKWARD]])
    if config.metrics.draw_graph:
        draw_pipeline_schedule(save_file=f'pipeline_schedule/{timestamp}_frozen_pipeline_schedule_rank{config.comm.local_rank}.svg',
                        config=config,
                        pipeline_schedule=pipeline_schedule_freezing,
                        title=f"Batch Time: {batch_time:.2f} ms (Average Freeze Ratio: {average_freeze_ratio:.2f})",
                        xlabel="Time (ms)", ylabel="Rank", tick_unit=200
                        )
    logger.info(f"\t> Batch Time: {batch_time:.2f} ms (Average Freeze Ratio: {average_freeze_ratio:.2f}, Time Reduction Rate: {1 - (batch_time / max_batch_time):.2f})")
    return pipeline_schedule_freezing


def adjust_freeze_ratio(pipeline_schedule:List[List[ActionWithFreezing]], monitored_values_dict: Dict[int, List[Tuple[float, float]]], config:TimelyFreezeConfig)-> List[List[ActionWithFreezing]]:
    '''
    Assume the given pipeline_schedule is of ActionWithFreezing type.
    Adjust the freeze ratio of each action in the pipeline schedule to minimize the batch time.
    This function will set the expected_freeze_ratio of each action based on the maximum and minimum batch time.
    It will also visualize the pipeline schedule with the adjusted freeze ratio.
    Arguments:
    - pipeline_schedule: List of List of ActionWithFreezing, the pipeline schedule to be adjusted.
    - monitored_values: Dict[stage] -> List of tuples. Each tuple is (freeze ratio, time) or (freeze ratio, time, microbatch). If microbatch is provided, points are colored by microbatch (larger → darker).
    Returns:
    - Updated pipeline_schedule with adjusted freeze ratio.
    '''

    timestamp = time.strftime("%y%m%d_%H%M")
    n_stages = config.parallelism.stages_per_rank
    if config.metrics.draw_graph:
        fig, axes = plt.subplots(1, n_stages, figsize=(3*n_stages, 3)) # 3 inches per plot, 3 inches height

    def draw_trend_line(stage: int, ax: plt.Axes, afrs: List[float], times: List[float], y_range: List[float], microbatches=None) -> Tuple[float, float]:
        ''' Draw the trend line for the monitored values.
        Arguments:
        - stage: the stage index for the title.
        - ax: the axis to draw the trend line on.
        - afrs: List of freeze ratios.
        - times: List of times corresponding to the freeze ratios.
        Returns:
        - Tuple of slope (a) and intercept (b) of the trend line.
        '''
        a, b = np.polyfit(afrs, times, 1)
        # assert a < 0, f"Expected a negative slope for the trend line. a: {a}, b: {b}"
        trend_fn = lambda r: a * r + b # linear function for the trend line
        ax.set_xlim(0, 1)
        y_min, y_max = y_range # min(times) * 0.9, max(times) * 1.1
        ax.set_ylim(y_min, y_max) # set the y limit to 10% above the maximum time
        r_range = np.linspace(0, 1, 100)
        t_range = trend_fn(r_range)
        ax.plot(r_range, t_range, linestyle='-', color='#685A3A', linewidth=1, label=f't = {a:.2f}r + {b:.2f}') # EDE8DF
        # color points by microbatch: higher microbatch → darker
        if microbatches is not None and len(microbatches) == len(times):
            mb_arr = np.array(microbatches, dtype=float)
            if mb_arr.size > 0:
                mb_min, mb_max = float(mb_arr.min()), float(mb_arr.max())
                if mb_max > mb_min:
                    norm = (mb_arr - mb_min) / (mb_max - mb_min)
                else:
                    norm = np.zeros_like(mb_arr)
                base = np.array([0xD3, 0xC4, 0xA5], dtype=float) / 255.0
                factors = 1.0 - 0.6 * norm  # 1.0 (light) → 0.4 (dark)
                colors = np.clip(base[None, :] * factors[:, None], 0.0, 1.0)
                ax.scatter(afrs, times, c=colors, marker='o', s=1)
            else:
                ax.scatter(afrs, times, color="#AE9B71", marker='o', s=1)
        else:
            ax.scatter(afrs, times, color='#AE9B71', marker='o', s=1)
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
        ax.set_yticks([y_min, y_min*0.75+y_max*0.25, (y_min+y_max)/2, y_min*0.25+y_max*0.75, y_max])
        ax.set_xlabel('Freeze Ratio', fontsize=9)
        ax.set_ylabel('Time (ms)', fontsize=9)
        ax.legend(loc='upper left', fontsize=9)
        ax.set_title(f'Stage {stage}', fontdict={'fontsize': 9})
        for spine in ax.spines.values(): # remove all boundary spines (left, right, top, bottom)
            spine.set_visible(False)
        return a, b

    durations_per_stage :torch.Tensor = torch.zeros((n_stages, 2), device=f'cuda:{config.comm.local_rank}')
    stages_order = {s:i for i,s in enumerate(sorted(set(monitored_values_dict.keys())))}
    # Accept both (afr, time) and (afr, time, microbatch)
    monitored_values_dict = {
        s: [
            [v[0] for v in monitored_values_dict[s]],
            [v[1] for v in monitored_values_dict[s]],
            [v[2] if isinstance(v, (list, tuple)) and len(v) > 2 else 0 for v in monitored_values_dict[s]]
        ]
        for s in stages_order.keys()
    }
    y_range_per_stage = {s: [min(monitored_values_dict[s][1]), max(monitored_values_dict[s][1])] for s in stages_order.keys()}

    for stage, i in stages_order.items():
        # trend line for of monitored values
        if config.metrics.draw_graph:
            axis = axes if len(stages_order) == 1 else axes[i]
            a, b = draw_trend_line(stage, axis, monitored_values_dict[stage][0], monitored_values_dict[stage][1], y_range_per_stage[stage], monitored_values_dict[stage][2])
        else:
            a, b = np.polyfit(monitored_values_dict[stage][0], monitored_values_dict[stage][1], 1)
        durations_per_stage[i][0] = b # max_duration (no freezing) = a * 0 + b
        durations_per_stage[i][1] = a + b # min_duration (all freezing) = a * 1 + b

    # draw the trend line for the entire rank schedule
    if config.metrics.draw_graph:
        title = 'Observed values (freeze ratio vs time) with Trend Line'
        # fig.suptitle(title, fontsize=12)
        plt.tight_layout()
        save_file = get_abs_path(f'pipeline_schedule_adjustment/{timestamp}_rank{config.comm.global_rank}_trend_line.svg', base_dir=config.metrics.image_folder)
        plt.savefig(save_file)
        plt.close()
        logger.info(f"{title} is saved as: {save_file}")


    durations :List[torch.Tensor] = [torch.zeros_like(durations_per_stage, device=f'cuda:{config.comm.local_rank}') for _ in range(config.comm.pp)]
    # gather trend lines across all pipelines
    dist.all_gather(durations, durations_per_stage, group=config.comm.pp_group) # gather the durations from all ranks
    durations = [d.cpu() for d in durations] # move the durations to CPU for further processing

    # set the max_duration and min_duration of each action in the pipeline schedule
    for r, rank_schedule in enumerate(pipeline_schedule):
        stages_order = {s:i for i,s in enumerate(sorted(set([action.stage for action in rank_schedule])))}
        for action in rank_schedule:
            if not action.freezable:
                continue
            action.max_duration = float(durations[r][stages_order[action.stage]][0])
            action.min_duration = float(durations[r][stages_order[action.stage]][1])

    # calculate the expected freeze ratio based on the maximum and minimum batch time and schedule the pipeline.
    max_freeze_ratio = getattr(config.freezing, "max_freeze_ratio", 0.9)
    pipeline_schedule = solve_dag_lp(pipeline_schedule, max_freeze_ratio=max_freeze_ratio) # solve the DAG LP problem to find the optimal schedule

    if config.comm.is_last_stage:
        batch_time = max([rank_actions[-1].end_time for rank_actions in pipeline_schedule])
        average_freeze_ratio = sum([action.expected_freeze_ratio for rank_actions in pipeline_schedule for action in rank_actions if action.type in [ActionType.BACKWARD_WEIGHT, ActionType.FULL_BACKWARD]]) / sum([1 for rank_actions in pipeline_schedule for action in rank_actions if action.type in [ActionType.BACKWARD_WEIGHT, ActionType.FULL_BACKWARD]])
        if config.metrics.draw_graph:
            draw_pipeline_schedule(save_file=f'pipeline_schedule/{timestamp}_adjusted_frozen_pipeline_schedule.svg',
                            config=config,
                            pipeline_schedule=pipeline_schedule,
                            title=f"Adjusted Batch Time: {batch_time:.2f} ms (Average Freeze Ratio: {average_freeze_ratio:.2f})",
                            xlabel="Time (ms)", ylabel="Rank", tick_unit=200
                            )
        logger.info(f"\t> Batch Time: {batch_time:.2f} ms (Average Freeze Ratio: {average_freeze_ratio:.2f})")
    return pipeline_schedule

