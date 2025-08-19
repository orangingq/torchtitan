import time
from matplotlib import pyplot as plt
import numpy as np
import torch
import torch.distributed as dist
from typing import List, Dict, Tuple
from scipy.optimize import linprog
import networkx as nx
from torchtitan.tools.logging import logger

from .action import Action, ActionType, ActionWithTime, ActionWithLog, ActionWithFreezing
from .util import draw_pipeline_schedule, get_abs_path
from .config import TimelyFreezeConfig, Comm

def gather_pipeline_schedule(log_schedule:List[ActionWithLog], CommConfig: Comm)-> List[List[ActionWithTime]]:
    ''' 
    Gather the pipeline schedule from all ranks and create a pipeline schedule with freezing actions.
    If the rank is not the last stage, it will return an empty list.
    '''

    # collect the number of actions per rank
    num_actions = len(log_schedule)
    num_actions_all = [torch.zeros((), device=f'cuda:{CommConfig.local_rank}', dtype=torch.int32) for _ in range(CommConfig.pp)]
    dist.all_gather(tensor_list=num_actions_all, tensor=torch.tensor(num_actions, device=f'cuda:{CommConfig.local_rank}', dtype=torch.int32), group=CommConfig.pp_group)
    num_actions_all = torch.stack(num_actions_all).cpu().tolist() # [num_pipelines]

    # create a tensor to send/gather from all ranks: (schedule info) x (num_actions)
    schedule_info = len(log_schedule[0].to_tensor(with_median=True)) # 5
    data_to_gather = torch.zeros(schedule_info, max(num_actions_all))
    data_to_gather[:, :num_actions] = torch.stack([action.to_tensor(with_median=True) for action in log_schedule]).T
    data_gather_list = [torch.zeros_like(data_to_gather, device=f'cuda:{CommConfig.local_rank}') for _ in range(CommConfig.pp)]
    dist.all_gather(tensor_list=data_gather_list, tensor=data_to_gather.to(CommConfig.local_rank), group=CommConfig.pp_group)
    data_gather_list = torch.stack(data_gather_list).cpu() # [num_pipelines, schedule, num_actions]

    # create a pipeline schedule in class ActionWithTime
    pipeline_schedule: List[List[ActionWithTime]] = [[ActionWithTime(*action.tolist()) for action in rank_list.T[:num_actions_all[r]]] for r, rank_list in enumerate(data_gather_list)] 
    return link_actions(pipeline_schedule)
    
def link_actions(pipeline_schedule:List[List[ActionWithTime]])-> List[List[ActionWithTime]]:
    '''Edge Construction
    Link the actions in the pipeline schedule based on the microbatch, stage, and action type.
    This function will set the prev_actions and next_actions attributes of each action.
    '''
    
    if any([len(action.prev_actions)>0 for actions_per_rank in pipeline_schedule for action in actions_per_rank]) \
        or any([len(action.next_actions)>0 for actions_per_rank in pipeline_schedule for action in actions_per_rank]):
            return pipeline_schedule # already linked, return the original schedule
        
    action_dict: Dict[tuple, ActionWithTime] = {(action.type, action.rank, action.microbatch, action.stage): action \
                                                    for actions_per_rank in pipeline_schedule for action in actions_per_rank}
    def set_links(prev, next):
        '''set the prev/next links between actions'''
        assert prev is not None and next is not None, f"prev and next should not be None. prev: {prev}, next: {next}"
        if next in prev.next_actions:
            assert prev in next.prev_actions, f"prev should be in next.prev_actions. prev: {prev}, next: {next}"
            return # already linked, do nothing
        prev.next_actions.append(next)
        next.prev_actions.append(prev)
        return 
    
    num_ranks: int = len(pipeline_schedule) 
    stages_per_rank = [list(dict.fromkeys([action.stage for action in pipeline_schedule[rank] if action.type == ActionType.FORWARD and action.stage is not None])) for rank in range(num_ranks)]
    last_stage: int = max([max(stages) for stages in stages_per_rank])
    num_microbatches: int = max([action.microbatch for action in pipeline_schedule[0]])+1

    # add prev/next links between actions
    for rank_actions in pipeline_schedule:
        for itr, action in enumerate(rank_actions):
            # next scheduled action
            if itr < len(rank_actions)-1: # if not the last action in the rank
                next_action = rank_actions[itr+1]
                set_links(action, next_action)
                
            # microbatch conditions
            if action.microbatch < num_microbatches-1: # microbatch n should be executed after microbatch n-1
                next_action = action_dict.get((action.type, action.rank, action.microbatch+1, action.stage), None)
                set_links(action, next_action)
            
            # action type conditions
            if action.type == ActionType.FORWARD: # full-backward / backward-input should be executed after forward
                next_action = action_dict.get((ActionType.FULL_BACKWARD, action.rank, action.microbatch, action.stage), 
                                action_dict.get((ActionType.BACKWARD_INPUT, action.rank, action.microbatch, action.stage), None))
                set_links(action, next_action)
            elif action.type == ActionType.BACKWARD_INPUT: # backward-weight should be executed after backward-input
                next_action = action_dict.get((ActionType.BACKWARD_WEIGHT, action.rank, action.microbatch, action.stage), None)
                set_links(action, next_action)
                
            # stage conditions
            if action.type == ActionType.FORWARD and action.stage < last_stage: # forward of stage n should be executed after stage n-1
                next_action = action_dict.get((action.type, (action.rank+1)%num_ranks, action.microbatch, action.stage+1),
                                action_dict.get((action.type, (action.rank-1+num_ranks)%num_ranks, action.microbatch, action.stage+1), None))             
                set_links(action, next_action)
            if action.type in [ActionType.FULL_BACKWARD, ActionType.BACKWARD_INPUT] and action.stage > 0: # full-backward / backward-input of stage n should be executed after stage n+1
                next_action = action_dict.get((ActionType.FULL_BACKWARD, (action.rank-1+num_ranks)%num_ranks, action.microbatch, action.stage-1), \
                                action_dict.get((ActionType.FULL_BACKWARD, (action.rank+1)%num_ranks, action.microbatch, action.stage-1), \
                                action_dict.get((ActionType.BACKWARD_INPUT, (action.rank-1+num_ranks)%num_ranks, action.microbatch, action.stage-1), \
                                action_dict.get((ActionType.BACKWARD_INPUT, (action.rank+1)%num_ranks, action.microbatch, action.stage-1), None))))
                set_links(action, next_action)
    return pipeline_schedule



def solve_dag_lp(pipeline_schedule:List[List[ActionWithFreezing]])-> List[List[ActionWithFreezing]]:
    '''
    Updated on July 29, 2025.
    Solve the DAG LP problem to find the optimal schedule for the pipeline schedule.
    This function assumes that each action in the pipeline schedule is all linked as a DAG and has max_duration and min_duration.
    This function returns the pipeline schedule with the start time and end time of each action set.
    '''
    start_node = ActionWithFreezing(ActionType.START, 0, 0, 0, 0) # dummy start node
    end_node = ActionWithFreezing(ActionType.FINISH, 0, 0, 0, 0) # dummy end node
    actions = [action for actions_per_rank in pipeline_schedule for action in actions_per_rank]
    actions = [start_node] + actions + [end_node] # add start and end nodes to the actions
    action_indices = {str(action): i for i, action in enumerate(actions)}
    n = len(actions) # number of actions including start and end nodes
    assert n == 2 + sum([len(rank_actions) for rank_actions in pipeline_schedule]), f"num_actions: {n}, sum of rank actions: {sum([len(rank_actions) for rank_actions in pipeline_schedule])}"
    G = nx.DiGraph() # directed graph to represent the pipeline schedule

    # add nodes
    for action in actions:
        G.add_node(str(action))
    
    # add edges
    for action in actions:
        if action.type != ActionType.START and len(action.prev_actions) == 0: # if the action is the first action in the rank, link it to the start node
            G.add_edge(str(start_node), str(action))
        if action.type != ActionType.FINISH and len(action.next_actions) == 0: # if the action is the last action in the rank, link it to the end node
            G.add_edge(str(action), str(end_node))
        for next_action in action.next_actions:
            G.add_edge(str(action), str(next_action))

    num_vars = 2 * n # action.start_time, action.duration
    c = np.zeros(num_vars) # cost vector for the linear programming problem
    c[n-1] = 9999.0 # first goal: minimize the total batch time
    for i in range(n): # second goal: maximize action durations (=minimize freeze ratio)
        action_type, w_min, w_max = actions[i].type, actions[i].min_duration, actions[i].max_duration
        if action_type in [ActionType.FULL_BACKWARD, ActionType.BACKWARD_WEIGHT] and w_min < w_max:
            c[n+i] = -1/(w_max - w_min)
        else:
            c[n+i] = 0

    A, b = [], [] # constraints for the linear programming problem
    # constraint of each edge u->v: u.start_time + u.duration <= v.start_time 
    for u, v in G.edges: # <=> (u.start_time - v.start_time + u.duration <= 0)
        row = np.zeros(num_vars) 
        row[action_indices[u]], row[action_indices[v]], row[n + action_indices[u]] = 1, -1, 1
        A.append(row)
        b.append(0)

    # constraint for the start node
    row = np.zeros(num_vars) 
    row[0], row[n] = 1, 1 # start_node.start_time + start_node.duration <= 0
    A.append(row)
    b.append(0)

    # Time duration bounds for each action
    bounds = [(0, None)] * n  # min bounds for start_time of each action (start_time >= 0)
    bounds += [(a.min_duration, a.max_duration) for a in actions] # bounds for the duration of each action

    res = linprog(c, A_ub=np.array(A), b_ub=np.array(b), bounds=bounds, method="highs")

    if res.success:
        start_times = res.x[:n]
        durations = res.x[n:2*n]
        for i, action in enumerate(actions):
            action.start_time = start_times[i]
            action.duration = durations[i]
    else:
        raise RuntimeError("Linear programming failed.")
    return pipeline_schedule


def schedule_pipeline(pipeline_schedule:List[List[ActionWithTime]],
        fwd_time:List[float]=None, bwd_time:List[float]=None, bwd_input_time:List[float]=None, bwd_weight_time:List[float]=None
    )-> List[List[ActionWithTime]]:
    '''
    Set the start time and end time of each action in the pipeline schedule.
    Update the duration of each action if fwd_time, bwd_time, bwd_input_time, bwd_weight_time are provided.
    '''
    num_ranks = len(pipeline_schedule) 
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
        draw_pipeline_schedule(save_file=f'{config.metrics.basename}/pipeline_schedule/{timestamp}_max_batch_time.svg',
                            pipeline_schedule=pipeline_schedule_freezing,
                            config=config,
                            title=f"Max Batch Time: {max_batch_time:.2f} ms",
                            xlabel="Time (ms)", ylabel="Rank"
                            )
        
    # initialize the duration of each action to have a minimum batch time
    pipeline_schedule_freezing = schedule_pipeline(set_expected_freeze_ratio(pipeline_schedule_freezing, ratio=1))
    min_batch_time = max([rank_actions[-1].end_time for rank_actions in pipeline_schedule_freezing]) # get the minimum batch time
    if config.comm.is_last_stage and config.metrics.draw_graph:
        draw_pipeline_schedule(save_file=f'{config.metrics.basename}/pipeline_schedule/{timestamp}_min_batch_time.svg',
                            pipeline_schedule=pipeline_schedule_freezing,
                            config=config,
                            title=f"Min Batch Time: {min_batch_time:.2f} ms",
                            xlabel="Time (ms)", ylabel="Rank"
                            )

    # calculate the expected freeze ratio based on the maximum and minimum batch time and schedule the pipeline.
    pipeline_schedule_freezing = solve_dag_lp(pipeline_schedule_freezing) # solve the DAG LP problem to find the optimal schedule

    if config.comm.is_last_stage:
        batch_time = max([rank_actions[-1].end_time for rank_actions in pipeline_schedule_freezing])
        average_freeze_ratio = sum([action.expected_freeze_ratio for rank_actions in pipeline_schedule_freezing for action in rank_actions if action.type in [ActionType.BACKWARD_WEIGHT, ActionType.FULL_BACKWARD]]) / sum([1 for rank_actions in pipeline_schedule_freezing for action in rank_actions if action.type in [ActionType.BACKWARD_WEIGHT, ActionType.FULL_BACKWARD]])
        if config.metrics.draw_graph:
            draw_pipeline_schedule(save_file=f'{config.metrics.basename}/pipeline_schedule/{timestamp}_frozen_pipeline_schedule.svg',
                            pipeline_schedule=pipeline_schedule_freezing,
                            config=config,
                            title=f"Batch Time: {batch_time:.2f} ms (Average Freeze Ratio: {average_freeze_ratio:.2f})",
                            xlabel="Time (ms)", ylabel="Rank"
                            )
        logger.info(f"> Batch Time: {batch_time:.2f} ms (Average Freeze Ratio: {average_freeze_ratio:.2f}, Time Reduction Rate: {1 - (batch_time / max_batch_time):.2f})", rank=False, timestamp='')
    return pipeline_schedule_freezing



def adjust_freeze_ratio(pipeline_schedule:List[List[ActionWithFreezing]], monitored_values_dict: Dict[int, List[Tuple[float, float]]], config: TimelyFreezeConfig)-> List[List[ActionWithFreezing]]:
    '''
    Assume the given pipeline_schedule is of ActionWithFreezing type.
    Adjust the freeze ratio of each action in the pipeline schedule to minimize the batch time.
    This function will set the expected_freeze_ratio of each action based on the maximum and minimum batch time.
    It will also visualize the pipeline schedule with the adjusted freeze ratio.
    Arguments:
    - pipeline_schedule: List of List of ActionWithFreezing, the pipeline schedule to be adjusted.
    - monitored_values: Dictionary of monitored values for each action, where the key is a tuple of the stage index and the value is a list of tuples of (freeze ratio, time).
    Returns:
    - Updated pipeline_schedule with adjusted freeze ratio.
    '''

    timestamp = time.strftime("%y%m%d_%H%M")
    n_stages = config.parallelism.stages_per_rank
    if config.metrics.draw_graph:
        fig, axes = plt.subplots(1, n_stages, figsize=(3*n_stages, 3)) # 3 inches per plot, 3 inches height

    def draw_trend_line(stage: int, ax: plt.Axes, afrs: List[float], times: List[float], y_range: List[float]) -> Tuple[float, float]:
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
        ax.scatter(afrs, times, color='#AE9B71', marker='o', s=1) # 7A6A45 988456 A08A5A
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
        ax.set_yticks([y_min, y_min*0.75+y_max*0.25, (y_min+y_max)/2, y_min*0.25+y_max*0.75, y_max])
        ax.set_xlabel('Freeze Ratio', fontsize=9)
        ax.set_ylabel('Time (ms)', fontsize=9)
        ax.legend(loc='upper left', fontsize=9)
        ax.set_title(f'Stage {stage}', fontdict={'fontsize': 9})
        return a, b

    durations_per_stage :torch.Tensor = torch.zeros((n_stages, 2), device=f'cuda:{config.comm.local_rank}')
    stages_list = config.parallelism.stages_list
    monitored_values_dict = {s: [[v[0] for v in monitored_values_dict[s]], [v[1] for v in monitored_values_dict[s]]] for s in stages_list}
    y_range_per_stage = {s: [min(monitored_values_dict[s][1]), max(monitored_values_dict[s][1])] for s in stages_list}

    for i, stage in enumerate(stages_list):
        # trend line for of monitored values
        if config.metrics.draw_graph:
            axis = axes if len(stages_list) == 1 else axes[i]
            a, b = draw_trend_line(stage, axis, monitored_values_dict[stage][0], monitored_values_dict[stage][1], y_range_per_stage[stage])
        else:
            a, b = np.polyfit(monitored_values_dict[stage][0], monitored_values_dict[stage][1], 1)
        durations_per_stage[i][0] = b # max_duration (no freezing) = a * 0 + b
        durations_per_stage[i][1] = a + b # min_duration (all freezing) = a * 1 + b

    # draw the trend line for the entire rank schedule
    if config.metrics.draw_graph:
        title = 'Observed values (freeze ratio vs time) with Trend Line'
        # fig.suptitle(title, fontsize=12)
        plt.tight_layout()
        save_file = get_abs_path(f'{config.metrics.basename}/pipeline_schedule_adjustment/{timestamp}_rank{config.comm.global_rank}_trend_line.svg', 'image', make=True)
        plt.savefig(save_file)
        plt.close()
        logger.info(f"{title} is saved as: {save_file}")


    durations :List[torch.Tensor] = [torch.zeros_like(durations_per_stage, device=f'cuda:{config.comm.local_rank}') for _ in range(config.comm.pp)]
    # gather trend lines across all pipelines
    dist.all_gather(durations, durations_per_stage, group=config.comm.pp_group) # gather the durations from all ranks
    durations = [d.cpu() for d in durations] # move the durations to CPU for further processing

    # set the max_duration and min_duration of each action in the pipeline schedule
    for r, rank_schedule in enumerate(pipeline_schedule):
        stages_order = {s:i for i,s in enumerate(config.parallelism.stages_list)} # map stage to its order
        for action in rank_schedule:
            if action.type not in [ActionType.FULL_BACKWARD, ActionType.BACKWARD_WEIGHT]:
                continue
            action.max_duration = float(durations[r][stages_order[action.stage]][0])
            action.min_duration = float(durations[r][stages_order[action.stage]][1])

    # calculate the expected freeze ratio based on the maximum and minimum batch time and schedule the pipeline.
    pipeline_schedule = solve_dag_lp(pipeline_schedule) # solve the DAG LP problem to find the optimal schedule

    if config.comm.is_last_stage:
        batch_time = max([rank_actions[-1].end_time for rank_actions in pipeline_schedule])
        average_freeze_ratio = sum([action.expected_freeze_ratio for rank_actions in pipeline_schedule for action in rank_actions if action.type in [ActionType.BACKWARD_WEIGHT, ActionType.FULL_BACKWARD]]) / sum([1 for rank_actions in pipeline_schedule for action in rank_actions if action.type in [ActionType.BACKWARD_WEIGHT, ActionType.FULL_BACKWARD]])
        if config.metrics.draw_graph:
            draw_pipeline_schedule(save_file=f'{config.metrics.basename}/pipeline_schedule/{timestamp}_adjusted_frozen_pipeline_schedule.svg',
                            pipeline_schedule=pipeline_schedule,
                            config=config,
                            title=f"Adjusted Batch Time: {batch_time:.2f} ms (Average Freeze Ratio: {average_freeze_ratio:.2f})",
                            xlabel="Time (ms)", ylabel="Rank"
                            )
        logger.info(f"> Batch Time: {batch_time:.2f} ms (Average Freeze Ratio: {average_freeze_ratio:.2f})", rank=False, timestamp='')
    return pipeline_schedule


# if __name__ == "__main__":
#     # Test the draw_pipeline_schedule function
#     from .util import draw_pipeline_schedule
#     from .config import global_config

#     # # Example pipeline schedule 1 : easy example
#     # pipeline_schedule = [
#     #     [ActionWithTime(ActionType.FORWARD, 0, 0), ActionWithTime(ActionType.FULL_BACKWARD, 0, 0)],
#     #     [ActionWithTime(ActionType.FORWARD, 1, 0), ActionWithTime(ActionType.FULL_BACKWARD, 1, 0)],
#     #     [ActionWithTime(ActionType.FORWARD, 2, 0), ActionWithTime(ActionType.FULL_BACKWARD, 2, 0)]
#     # ]
#     # fwd_time = [1.0, 1.5, 2.0]
#     # bwd_time = [1.5, 2.0, 2.5]
#     # pipeline_schedule = schedule_pipeline(pipeline_schedule, fwd_time, bwd_time)
#     # draw_pipeline_schedule("pipeline_schedule.pdf", pipeline_schedule)
#     # pipeline_schedule_freezing = set_freeze_ratio(pipeline_schedule)
#     # draw_pipeline_schedule("pipeline_schedule_frozen.pdf", pipeline_schedule_freezing)

#     # Example pipeline schedule 2 : interleavedzb, 4 ranks, 8 microbatches, 2 stages per rank
#     # 2-1 : Realistic version
#     global_config.parallelism.bwd_separated = True # set the backward separated mode
#     pipeline_schedule = [
#         [ActionWithTime(1, 0, 0, 0), ActionWithTime(1, 0, 1, 0), ActionWithTime(1, 0, 2, 0), ActionWithTime(1, 0, 3, 0), ActionWithTime(1, 0, 0, 4), ActionWithTime(1, 0, 1, 4), ActionWithTime(1, 0, 2, 4), ActionWithTime(1, 0, 3, 4), ActionWithTime(2, 0, 0, 4), ActionWithTime(3, 0, 0, 4), ActionWithTime(1, 0, 4, 0), ActionWithTime(2, 0, 1, 4), ActionWithTime(3, 0, 1, 4), ActionWithTime(1, 0, 5, 0), ActionWithTime(2, 0, 2, 4), ActionWithTime(3, 0, 2, 4), ActionWithTime(1, 0, 6, 0), ActionWithTime(2, 0, 3, 4), ActionWithTime(3, 0, 3, 4), ActionWithTime(1, 0, 7, 0), ActionWithTime(10, 0, 0, 0), ActionWithTime(1, 0, 4, 4), ActionWithTime(10, 0, 1, 0), ActionWithTime(1, 0, 5, 4), ActionWithTime(10, 0, 2, 0), ActionWithTime(1, 0, 6, 4), ActionWithTime(10, 0, 3, 0), ActionWithTime(1, 0, 7, 4), ActionWithTime(2, 0, 4, 4), ActionWithTime(3, 0, 4, 4), ActionWithTime(2, 0, 5, 4), ActionWithTime(3, 0, 5, 4), ActionWithTime(2, 0, 6, 4), ActionWithTime(3, 0, 6, 4), ActionWithTime(2, 0, 7, 4), ActionWithTime(3, 0, 7, 4), ActionWithTime(10, 0, 4, 0), ActionWithTime(10, 0, 5, 0), ActionWithTime(10, 0, 6, 0), ActionWithTime(10, 0, 7, 0)],
#         [ActionWithTime(1, 1, 0, 1), ActionWithTime(1, 1, 1, 1), ActionWithTime(1, 1, 2, 1), ActionWithTime(1, 1, 3, 1), ActionWithTime(1, 1, 0, 5), ActionWithTime(1, 1, 1, 5), ActionWithTime(1, 1, 2, 5), ActionWithTime(2, 1, 0, 5), ActionWithTime(1, 1, 3, 5), ActionWithTime(2, 1, 1, 5), ActionWithTime(3, 1, 0, 5), ActionWithTime(1, 1, 4, 1), ActionWithTime(2, 1, 2, 5), ActionWithTime(3, 1, 1, 5), ActionWithTime(1, 1, 5, 1), ActionWithTime(2, 1, 3, 5), ActionWithTime(3, 1, 2, 5), ActionWithTime(1, 1, 6, 1), ActionWithTime(2, 1, 0, 1), ActionWithTime(3, 1, 3, 5), ActionWithTime(1, 1, 7, 1), ActionWithTime(2, 1, 1, 1), ActionWithTime(3, 1, 0, 1), ActionWithTime(1, 1, 4, 5), ActionWithTime(2, 1, 2, 1), ActionWithTime(3, 1, 1, 1), ActionWithTime(1, 1, 5, 5), ActionWithTime(2, 1, 3, 1), ActionWithTime(3, 1, 2, 1), ActionWithTime(1, 1, 6, 5), ActionWithTime(2, 1, 4, 5), ActionWithTime(3, 1, 3, 1), ActionWithTime(1, 1, 7, 5), ActionWithTime(2, 1, 5, 5), ActionWithTime(3, 1, 4, 5), ActionWithTime(2, 1, 6, 5), ActionWithTime(3, 1, 5, 5), ActionWithTime(2, 1, 7, 5), ActionWithTime(3, 1, 6, 5), ActionWithTime(2, 1, 4, 1), ActionWithTime(3, 1, 7, 5), ActionWithTime(2, 1, 5, 1), ActionWithTime(3, 1, 4, 1), ActionWithTime(2, 1, 6, 1), ActionWithTime(3, 1, 5, 1), ActionWithTime(2, 1, 7, 1), ActionWithTime(3, 1, 6, 1), ActionWithTime(3, 1, 7, 1)], 
#         [ActionWithTime(1, 2, 0, 2), ActionWithTime(1, 2, 1, 2), ActionWithTime(1, 2, 2, 2), ActionWithTime(1, 2, 3, 2), ActionWithTime(1, 2, 0, 6), ActionWithTime(1, 2, 1, 6), ActionWithTime(2, 2, 0, 6), ActionWithTime(1, 2, 2, 6), ActionWithTime(2, 2, 1, 6), ActionWithTime(1, 2, 3, 6), ActionWithTime(2, 2, 2, 6), ActionWithTime(3, 2, 0, 6), ActionWithTime(1, 2, 4, 2), ActionWithTime(2, 2, 3, 6), ActionWithTime(3, 2, 1, 6), ActionWithTime(1, 2, 5, 2), ActionWithTime(2, 2, 0, 2), ActionWithTime(3, 2, 2, 6), ActionWithTime(1, 2, 6, 2), ActionWithTime(2, 2, 1, 2), ActionWithTime(3, 2, 3, 6), ActionWithTime(1, 2, 7, 2), ActionWithTime(2, 2, 2, 2), ActionWithTime(3, 2, 0, 2), ActionWithTime(1, 2, 4, 6), ActionWithTime(2, 2, 3, 2), ActionWithTime(3, 2, 1, 2), ActionWithTime(1, 2, 5, 6), ActionWithTime(2, 2, 4, 6), ActionWithTime(3, 2, 2, 2), ActionWithTime(1, 2, 6, 6), ActionWithTime(2, 2, 5, 6), ActionWithTime(3, 2, 3, 2), ActionWithTime(1, 2, 7, 6), ActionWithTime(2, 2, 6, 6), ActionWithTime(3, 2, 4, 6), ActionWithTime(2, 2, 7, 6), ActionWithTime(3, 2, 5, 6), ActionWithTime(2, 2, 4, 2), ActionWithTime(3, 2, 6, 6), ActionWithTime(2, 2, 5, 2), ActionWithTime(3, 2, 7, 6), ActionWithTime(2, 2, 6, 2), ActionWithTime(3, 2, 4, 2), ActionWithTime(2, 2, 7, 2), ActionWithTime(3, 2, 5, 2), ActionWithTime(3, 2, 6, 2), ActionWithTime(3, 2, 7, 2)], 
#         [ActionWithTime(1, 3, 0, 3), ActionWithTime(1, 3, 1, 3), ActionWithTime(1, 3, 2, 3), ActionWithTime(1, 3, 3, 3), ActionWithTime(1, 3, 0, 7), ActionWithTime(2, 3, 0, 7), ActionWithTime(1, 3, 1, 7), ActionWithTime(2, 3, 1, 7), ActionWithTime(1, 3, 2, 7), ActionWithTime(2, 3, 2, 7), ActionWithTime(1, 3, 3, 7), ActionWithTime(2, 3, 3, 7), ActionWithTime(3, 3, 0, 7), ActionWithTime(1, 3, 4, 3), ActionWithTime(2, 3, 0, 3), ActionWithTime(3, 3, 1, 7), ActionWithTime(1, 3, 5, 3), ActionWithTime(2, 3, 1, 3), ActionWithTime(3, 3, 2, 7), ActionWithTime(1, 3, 6, 3), ActionWithTime(2, 3, 2, 3), ActionWithTime(3, 3, 3, 7), ActionWithTime(1, 3, 7, 3), ActionWithTime(2, 3, 3, 3), ActionWithTime(3, 3, 0, 3), ActionWithTime(1, 3, 4, 7), ActionWithTime(2, 3, 4, 7), ActionWithTime(3, 3, 1, 3), ActionWithTime(1, 3, 5, 7), ActionWithTime(2, 3, 5, 7), ActionWithTime(3, 3, 2, 3), ActionWithTime(1, 3, 6, 7), ActionWithTime(2, 3, 6, 7), ActionWithTime(3, 3, 3, 3), ActionWithTime(1, 3, 7, 7), ActionWithTime(2, 3, 7, 7), ActionWithTime(3, 3, 4, 7), ActionWithTime(2, 3, 4, 3), ActionWithTime(3, 3, 5, 7), ActionWithTime(2, 3, 5, 3), ActionWithTime(3, 3, 6, 7), ActionWithTime(2, 3, 6, 3), ActionWithTime(3, 3, 7, 7), ActionWithTime(2, 3, 7, 3), ActionWithTime(3, 3, 4, 3), ActionWithTime(3, 3, 5, 3), ActionWithTime(3, 3, 6, 3), ActionWithTime(3, 3, 7, 3)]]
#     fwd_time = [7.12, 6.61, 6.65, 6.67, 7.12, 6.61, 6.65, 6.67]
#     bwd_time = [15.1, 0, 0, 0, 15.1, 0, 0, 0]
#     bwd_input_time = [10.7, 8.39, 10.41, 8.4, 10.7, 8.39, 10.41, 8.4]
#     bwd_weight_time = [3.96, 4.79, 3.84, 4.94, 3.96, 4.79, 3.84, 4.94]
#     pipeline_schedule = schedule_pipeline(pipeline_schedule, fwd_time, bwd_time, bwd_input_time, bwd_weight_time)
#     draw_pipeline_schedule("pipeline_schedule.pdf", pipeline_schedule)
#     pipeline_schedule_freezing = set_freeze_ratio(pipeline_schedule, global_config)
#     # batch_time = max([rank_actions[-1].end_time for rank_actions in pipeline_schedule_freezing])
#     # average_freeze_ratio = sum([action.expected_freeze_ratio for rank_actions in pipeline_schedule_freezing for action in rank_actions if action.type in [ActionType.BACKWARD_WEIGHT, ActionType.FULL_BACKWARD]]) / sum([1 for rank_actions in pipeline_schedule_freezing for action in rank_actions if action.type in [ActionType.BACKWARD_WEIGHT, ActionType.FULL_BACKWARD]])
#     # draw_pipeline_schedule(f"pipeline_schedule_frozen.svg", pipeline_schedule_freezing, title=f"Batch Time: {batch_time:.2f} ms (Average Freeze Ratio: {average_freeze_ratio:.2f})")
    
    

#     # 2-2 : uniform time version
#     uniform_time = [10,10,10,10,10,10,10,10]
#     for actions_per_rank in pipeline_schedule:
#         for action in actions_per_rank:
#             action.scheduled_flag = False # reset the schedule flag
#     pipeline_schedule = schedule_pipeline(pipeline_schedule, uniform_time, [20,20,20,20,20,20,20,20], uniform_time, uniform_time)
#     draw_pipeline_schedule("pipeline_schedule.pdf", pipeline_schedule)
#     pipeline_schedule_freezing = set_freeze_ratio(pipeline_schedule, global_config)
#     # batch_time = max([rank_actions[-1].end_time for rank_actions in pipeline_schedule_freezing])
#     # average_freeze_ratio = sum([action.expected_freeze_ratio for rank_actions in pipeline_schedule_freezing for action in rank_actions if action.type in [ActionType.BACKWARD_WEIGHT, ActionType.FULL_BACKWARD]]) / sum([1 for rank_actions in pipeline_schedule_freezing for action in rank_actions if action.type in [ActionType.BACKWARD_WEIGHT, ActionType.FULL_BACKWARD]])
#     # draw_pipeline_schedule(f"pipeline_schedule_frozen.svg", pipeline_schedule_freezing, title=f"Batch Time: {batch_time:.2f} ms (Average Freeze Ratio: {average_freeze_ratio:.2f})")
    
#     # Example pipeline schedule 3 : zbv, 4 ranks, 8 microbatches, 2 stages per rank
#     pipeline_schedule = [[ActionWithTime(ActionType.FORWARD, 0, 0, 0), ActionWithTime(ActionType.FORWARD, 0, 1, 0), ActionWithTime(ActionType.FORWARD, 0, 2, 0), ActionWithTime(ActionType.FORWARD, 0, 3, 0), ActionWithTime(ActionType.FORWARD, 0, 4, 0), ActionWithTime(ActionType.FORWARD, 0, 5, 0), ActionWithTime(ActionType.FORWARD, 0, 6, 0), ActionWithTime(ActionType.FORWARD, 0, 0, 3), ActionWithTime(ActionType.BACKWARD_INPUT, 0, 0, 3), ActionWithTime(ActionType.BACKWARD_WEIGHT, 0, 0, 3), ActionWithTime(ActionType.FORWARD, 0, 1, 3), ActionWithTime(ActionType.BACKWARD_INPUT, 0, 1, 3), ActionWithTime(ActionType.BACKWARD_WEIGHT, 0, 1, 3), ActionWithTime(ActionType.FORWARD, 0, 2, 3), ActionWithTime(ActionType.BACKWARD_INPUT, 0, 2, 3), ActionWithTime(ActionType.BACKWARD_WEIGHT, 0, 2, 3), ActionWithTime(ActionType.FORWARD, 0, 3, 3), ActionWithTime(ActionType.BACKWARD_INPUT, 0, 3, 3), ActionWithTime(ActionType.BACKWARD_WEIGHT, 0, 3, 3), ActionWithTime(ActionType.FORWARD, 0, 7, 0), ActionWithTime(ActionType.FULL_BACKWARD, 0, 0, 0), ActionWithTime(ActionType.FORWARD, 0, 4, 3), ActionWithTime(ActionType.BACKWARD_INPUT, 0, 4, 3), ActionWithTime(ActionType.BACKWARD_WEIGHT, 0, 4, 3), ActionWithTime(ActionType.FULL_BACKWARD, 0, 1, 0), ActionWithTime(ActionType.FORWARD, 0, 5, 3), ActionWithTime(ActionType.BACKWARD_INPUT, 0, 5, 3), ActionWithTime(ActionType.BACKWARD_WEIGHT, 0, 5, 3), ActionWithTime(ActionType.FULL_BACKWARD, 0, 2, 0), ActionWithTime(ActionType.FORWARD, 0, 6, 3), ActionWithTime(ActionType.BACKWARD_INPUT, 0, 6, 3), ActionWithTime(ActionType.BACKWARD_WEIGHT, 0, 6, 3), ActionWithTime(ActionType.FULL_BACKWARD, 0, 3, 0), ActionWithTime(ActionType.FORWARD, 0, 7, 3), ActionWithTime(ActionType.BACKWARD_INPUT, 0, 7, 3), ActionWithTime(ActionType.BACKWARD_WEIGHT, 0, 7, 3), ActionWithTime(ActionType.FULL_BACKWARD, 0, 4, 0), ActionWithTime(ActionType.FULL_BACKWARD, 0, 5, 0), ActionWithTime(ActionType.FULL_BACKWARD, 0, 6, 0), ActionWithTime(ActionType.FULL_BACKWARD, 0, 7, 0)], 
#                          [ActionWithTime(ActionType.FORWARD, 1, 0, 1), ActionWithTime(ActionType.FORWARD, 1, 1, 1), ActionWithTime(ActionType.FORWARD, 1, 2, 1), ActionWithTime(ActionType.FORWARD, 1, 3, 1), ActionWithTime(ActionType.FORWARD, 1, 4, 1), ActionWithTime(ActionType.FORWARD, 1, 0, 2), ActionWithTime(ActionType.FORWARD, 1, 5, 1), ActionWithTime(ActionType.FORWARD, 1, 1, 2), ActionWithTime(ActionType.BACKWARD_INPUT, 1, 0, 2), ActionWithTime(ActionType.BACKWARD_WEIGHT, 1, 0, 2), ActionWithTime(ActionType.FORWARD, 1, 2, 2), ActionWithTime(ActionType.BACKWARD_INPUT, 1, 1, 2), ActionWithTime(ActionType.BACKWARD_WEIGHT, 1, 1, 2), ActionWithTime(ActionType.FORWARD, 1, 3, 2), ActionWithTime(ActionType.BACKWARD_INPUT, 1, 2, 2), ActionWithTime(ActionType.BACKWARD_WEIGHT, 1, 2, 2), ActionWithTime(ActionType.FORWARD, 1, 6, 1), ActionWithTime(ActionType.BACKWARD_INPUT, 1, 0, 1), ActionWithTime(ActionType.BACKWARD_WEIGHT, 1, 0, 1), ActionWithTime(ActionType.FORWARD, 1, 4, 2), ActionWithTime(ActionType.BACKWARD_INPUT, 1, 3, 2), ActionWithTime(ActionType.BACKWARD_WEIGHT, 1, 3, 2), ActionWithTime(ActionType.FORWARD, 1, 7, 1), ActionWithTime(ActionType.BACKWARD_INPUT, 1, 1, 1), ActionWithTime(ActionType.BACKWARD_WEIGHT, 1, 1, 1), ActionWithTime(ActionType.FORWARD, 1, 5, 2), ActionWithTime(ActionType.BACKWARD_INPUT, 1, 4, 2), ActionWithTime(ActionType.BACKWARD_WEIGHT, 1, 4, 2), ActionWithTime(ActionType.BACKWARD_INPUT, 1, 2, 1), ActionWithTime(ActionType.BACKWARD_WEIGHT, 1, 2, 1), ActionWithTime(ActionType.FORWARD, 1, 6, 2), ActionWithTime(ActionType.BACKWARD_INPUT, 1, 5, 2), ActionWithTime(ActionType.BACKWARD_WEIGHT, 1, 5, 2), ActionWithTime(ActionType.BACKWARD_INPUT, 1, 3, 1), ActionWithTime(ActionType.BACKWARD_WEIGHT, 1, 3, 1), ActionWithTime(ActionType.FORWARD, 1, 7, 2), ActionWithTime(ActionType.BACKWARD_INPUT, 1, 6, 2), ActionWithTime(ActionType.BACKWARD_WEIGHT, 1, 6, 2), ActionWithTime(ActionType.BACKWARD_INPUT, 1, 4, 1), ActionWithTime(ActionType.BACKWARD_INPUT, 1, 7, 2), ActionWithTime(ActionType.BACKWARD_INPUT, 1, 5, 1), ActionWithTime(ActionType.BACKWARD_WEIGHT, 1, 4, 1), ActionWithTime(ActionType.BACKWARD_INPUT, 1, 6, 1), ActionWithTime(ActionType.BACKWARD_WEIGHT, 1, 5, 1), ActionWithTime(ActionType.BACKWARD_INPUT, 1, 7, 1), ActionWithTime(ActionType.BACKWARD_WEIGHT, 1, 6, 1), ActionWithTime(ActionType.BACKWARD_WEIGHT, 1, 7, 2), ActionWithTime(ActionType.BACKWARD_WEIGHT, 1, 7, 1)], 
#                          [ActionWithTime(ActionType.FORWARD, 2, 0, 2), ActionWithTime(ActionType.FORWARD, 2, 1, 2), ActionWithTime(ActionType.FORWARD, 2, 2, 2), ActionWithTime(ActionType.FORWARD, 2, 0, 1), ActionWithTime(ActionType.FORWARD, 2, 3, 2), ActionWithTime(ActionType.FORWARD, 2, 1, 1), ActionWithTime(ActionType.FORWARD, 2, 4, 2), ActionWithTime(ActionType.FORWARD, 2, 2, 1), ActionWithTime(ActionType.BACKWARD_INPUT, 2, 0, 1), ActionWithTime(ActionType.BACKWARD_WEIGHT, 2, 0, 1), ActionWithTime(ActionType.FORWARD, 2, 3, 1), ActionWithTime(ActionType.BACKWARD_INPUT, 2, 1, 1), ActionWithTime(ActionType.BACKWARD_WEIGHT, 2, 1, 1), ActionWithTime(ActionType.FORWARD, 2, 5, 2), ActionWithTime(ActionType.BACKWARD_INPUT, 2, 0, 2), ActionWithTime(ActionType.BACKWARD_WEIGHT, 2, 0, 2), ActionWithTime(ActionType.FORWARD, 2, 4, 1), ActionWithTime(ActionType.BACKWARD_INPUT, 2, 2, 1), ActionWithTime(ActionType.BACKWARD_WEIGHT, 2, 2, 1), ActionWithTime(ActionType.FORWARD, 2, 6, 2), ActionWithTime(ActionType.BACKWARD_INPUT, 2, 1, 2), ActionWithTime(ActionType.BACKWARD_WEIGHT, 2, 1, 2), ActionWithTime(ActionType.FORWARD, 2, 5, 1), ActionWithTime(ActionType.BACKWARD_INPUT, 2, 3, 1), ActionWithTime(ActionType.BACKWARD_WEIGHT, 2, 3, 1), ActionWithTime(ActionType.FORWARD, 2, 7, 2), ActionWithTime(ActionType.BACKWARD_INPUT, 2, 2, 2), ActionWithTime(ActionType.BACKWARD_WEIGHT, 2, 2, 2), ActionWithTime(ActionType.FORWARD, 2, 6, 1), ActionWithTime(ActionType.BACKWARD_INPUT, 2, 4, 1), ActionWithTime(ActionType.BACKWARD_WEIGHT, 2, 4, 1), ActionWithTime(ActionType.BACKWARD_INPUT, 2, 3, 2), ActionWithTime(ActionType.BACKWARD_WEIGHT, 2, 3, 2), ActionWithTime(ActionType.FORWARD, 2, 7, 1), ActionWithTime(ActionType.BACKWARD_INPUT, 2, 5, 1), ActionWithTime(ActionType.BACKWARD_WEIGHT, 2, 5, 1), ActionWithTime(ActionType.BACKWARD_INPUT, 2, 4, 2), ActionWithTime(ActionType.BACKWARD_INPUT, 2, 6, 1), ActionWithTime(ActionType.BACKWARD_INPUT, 2, 5, 2), ActionWithTime(ActionType.BACKWARD_INPUT, 2, 7, 1), ActionWithTime(ActionType.BACKWARD_INPUT, 2, 6, 2), ActionWithTime(ActionType.BACKWARD_WEIGHT, 2, 4, 2), ActionWithTime(ActionType.BACKWARD_INPUT, 2, 7, 2), ActionWithTime(ActionType.BACKWARD_WEIGHT, 2, 5, 2), ActionWithTime(ActionType.BACKWARD_WEIGHT, 2, 6, 1), ActionWithTime(ActionType.BACKWARD_WEIGHT, 2, 7, 1), ActionWithTime(ActionType.BACKWARD_WEIGHT, 2, 6, 2), ActionWithTime(ActionType.BACKWARD_WEIGHT, 2, 7, 2)], 
#                          [ActionWithTime(ActionType.FORWARD, 3, 0, 3), ActionWithTime(ActionType.FORWARD, 3, 0, 0), ActionWithTime(ActionType.FORWARD, 3, 1, 3), ActionWithTime(ActionType.FORWARD, 3, 1, 0), ActionWithTime(ActionType.FORWARD, 3, 2, 3), ActionWithTime(ActionType.FORWARD, 3, 2, 0), ActionWithTime(ActionType.FORWARD, 3, 3, 3), ActionWithTime(ActionType.FORWARD, 3, 3, 0), ActionWithTime(ActionType.FULL_BACKWARD, 3, 0, 0), ActionWithTime(ActionType.FORWARD, 3, 4, 3), ActionWithTime(ActionType.BACKWARD_INPUT, 3, 0, 3), ActionWithTime(ActionType.BACKWARD_WEIGHT, 3, 0, 3), ActionWithTime(ActionType.FORWARD, 3, 4, 0), ActionWithTime(ActionType.FULL_BACKWARD, 3, 1, 0), ActionWithTime(ActionType.FORWARD, 3, 5, 3), ActionWithTime(ActionType.BACKWARD_INPUT, 3, 1, 3), ActionWithTime(ActionType.BACKWARD_WEIGHT, 3, 1, 3), ActionWithTime(ActionType.FORWARD, 3, 5, 0), ActionWithTime(ActionType.FULL_BACKWARD, 3, 2, 0), ActionWithTime(ActionType.FORWARD, 3, 6, 3), ActionWithTime(ActionType.BACKWARD_INPUT, 3, 2, 3), ActionWithTime(ActionType.BACKWARD_WEIGHT, 3, 2, 3), ActionWithTime(ActionType.FORWARD, 3, 6, 0), ActionWithTime(ActionType.FULL_BACKWARD, 3, 3, 0), ActionWithTime(ActionType.FORWARD, 3, 7, 3), ActionWithTime(ActionType.BACKWARD_INPUT, 3, 3, 3), ActionWithTime(ActionType.BACKWARD_WEIGHT, 3, 3, 3), ActionWithTime(ActionType.FORWARD, 3, 7, 0), ActionWithTime(ActionType.FULL_BACKWARD, 3, 4, 0), ActionWithTime(ActionType.BACKWARD_INPUT, 3, 4, 3), ActionWithTime(ActionType.BACKWARD_INPUT, 3, 5, 3), ActionWithTime(ActionType.BACKWARD_INPUT, 3, 6, 3), ActionWithTime(ActionType.BACKWARD_INPUT, 3, 7, 3), ActionWithTime(ActionType.BACKWARD_WEIGHT, 3, 4, 3), ActionWithTime(ActionType.FULL_BACKWARD, 3, 5, 0), ActionWithTime(ActionType.FULL_BACKWARD, 3, 6, 0), ActionWithTime(ActionType.FULL_BACKWARD, 3, 7, 0), ActionWithTime(ActionType.BACKWARD_WEIGHT, 3, 5, 3), ActionWithTime(ActionType.BACKWARD_WEIGHT, 3, 6, 3), ActionWithTime(ActionType.BACKWARD_WEIGHT, 3, 7, 3)]]
#     fwd_time, bwd_time, bwd_input_time, bwd_weight_time \
#         = [8.3101, 10.3417,  9.5100,  7.3098], [14.8432,  0.0000,  0.0000, 13.5128], [11.4721, 16.9559, 10.7292,  9.1316], [10.2950, 26.8687, 15.5081,  7.1363]
#     pipeline_schedule = schedule_pipeline(pipeline_schedule, fwd_time, bwd_time, bwd_input_time, bwd_weight_time)
#     draw_pipeline_schedule("pipeline_schedule.pdf", pipeline_schedule)
#     pipeline_schedule_freezing = set_freeze_ratio(pipeline_schedule, global_config)
#     # batch_time = max([rank_actions[-1].end_time for rank_actions in pipeline_schedule_freezing])
#     # average_freeze_ratio = sum([action.expected_freeze_ratio for rank_actions in pipeline_schedule_freezing for action in rank_actions if action.type in [ActionType.BACKWARD_WEIGHT, ActionType.FULL_BACKWARD]]) / sum([1 for rank_actions in pipeline_schedule_freezing for action in rank_actions if action.type in [ActionType.BACKWARD_WEIGHT, ActionType.FULL_BACKWARD]])
#     # draw_pipeline_schedule(f"pipeline_schedule_frozen.svg", pipeline_schedule_freezing, title=f"Batch Time: {batch_time:.2f} ms (Average Freeze Ratio: {average_freeze_ratio:.2f})")
    