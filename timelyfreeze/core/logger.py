from enum import Enum
from typing import Dict, List, Union
import numpy as np
import torch
import timeit

from .util import log_time
from .action import Action, ActionType, ActionWithLog, ActionWithFreezing
from .config import global_config

class Range(str, Enum):
    START   = 'start'
    END     = 'end'
    NONE    = None

    @classmethod
    def all(cls) -> List[str]:
        return [r.value for r in cls]


class ActionPhase:
    START   : List[ActionType] = [ActionType.START]
    FORWARD : List[ActionType] = [ActionType.FORWARD]
    BACKWARD: List[ActionType] = [ActionType.FULL_BACKWARD]
    END     : List[ActionType] = [ActionType.SYNC]
    
    @classmethod
    def ALL(cls) -> List[ActionType]:
        '''Return all action types grouped by phase'''
        return cls.START + cls.FORWARD + cls.BACKWARD + cls.END


class RankSchedule:
    '''Rank schedule for pipeline parallelism.'''
    def __init__(self):
        self.schedule: List[Action] = [] # the schedule of actions, which is a list of Action
        self.action_dict: Dict[Tuple[ActionType, int, int], int] = {} # the action dict, which is a dict of index of the Action in self.schedule with key (step, pp_rank, microbatch, stage)

    def schedule_idx(self, step: ActionType, microbatch: int, stage: int) -> int:
        '''get the index of the action in the schedule'''
        return self.action_dict.get((step, microbatch, stage), -1)
    
    def add_action(self, action: Action) -> int:
        '''add an action to the schedule'''
        if action not in self.schedule:
            self.schedule.append(action)
            self.action_dict[(action.step, action.microbatch, action.stage)] = len(self.schedule) - 1
            return len(self.schedule) - 1
        else:
            return self.action_dict[(action.step, action.microbatch, action.stage)]

    def __len__(self) -> int:
        '''get the length of the schedule'''
        return len(self.schedule)
    def __getitem__(self, idx: int) -> Action:
        '''get the action at the index'''
        return self.schedule[idx]
    def __iter__(self):
        '''iterate over the schedule'''
        return iter(self.schedule)

class PipelineLogger:
    '''logger for the pipeline step.'''
    def __init__(self) -> None:
        self.disabled = False # temporary disable flag. stop logging if set to True
        self.is_first_cycle = True
        self.step_cnt = 0
        self.log_freq = 20 # the frequency of logging cuda time, in steps
        
        if global_config.parallelism.bwd_separated:
            ActionPhase.BACKWARD = [ActionType.FULL_BACKWARD, ActionType.BACKWARD_INPUT, ActionType.BACKWARD_WEIGHT]

        # log the schedule order
        self.rank_schedule = RankSchedule() # the rank schedule, which is a list of Action
        self.action_dict: Dict[Tuple[ActionType, int, int, int], ActionWithFreezing] = None # the action dict, which is a dict of ActionWithFreezing with key (step, pp_rank, microbatch, stage)

        # log time
        self.log_batch_time = []
        self.actions_list = {
            ActionType.FORWARD: {}, 
            ActionType.FULL_BACKWARD: {},
            ActionType.BACKWARD_INPUT: {},
            ActionType.BACKWARD_WEIGHT: {},
        }

        # cuda timer
        self.cuda_timer_batch = {Range.START: [], Range.END: []}
        self.cuda_timer_schedule = {Range.START: [], Range.END: []}
        self.timer_reset()
        return

    def initialize(self):
        '''initialize the log'''
        if args.bwd_separated:
            ActionPhase.BACKWARD = [ActionType.RECV_B, ActionType.FULL_BACKWARD, ActionType.BACKWARD_INPUT, ActionType.BACKWARD_WEIGHT, ActionType.SEND_B]
        self.step_cnt = 0
        self.timer_reset()
        return self
    
    def timer_reset(self):
        '''reset the timer'''
        self.cuda_timer_batch = {Range.START: [], Range.END: []}
        self.cuda_timer_schedule = {Range.START: [], Range.END: []}
        return

    def disable(self):
        '''disable the log (for validation)'''
        self.disabled = True
        return

    def enable(self):
        '''enable the log (for training)'''
        self.disabled = False
        return

    def __call__(self, microbatch:int, stage:int, step:ActionType, range:Range, postfix:str='', timestamp=None) -> 'PipelineLogger':
        '''log the pipeline step.
        Args:
            microbatch: int, the microbatch index
            step: Union[ActionType, str], the step name or ActionType number
            range: ['start', 'end', 'None']. If 'start', push the nvtx range. If 'end', pop the nvtx range. If 'None', mark the time point. 
            postfix: str, the postfix of the step
            timestamp: float, the timestamp of the step
        '''
        if self.disabled:
            return self
        assert step in ActionPhase.ALL(), f"step {step} should be in {ActionPhase.ALL()}"
        assert range in Range.ALL(), f"range should be one of {Range.ALL()}"
        assert 0 <= microbatch < global_config.parallelism.microbatches, f"microbatch {microbatch} should be less than {global_config.parallelism.microbatches}"

        # Process to identify pipeline schedule cycle (for the first cycle)
        schedule_idx :int = self.rank_schedule.schedule_idx(step, microbatch, stage)
        if self.is_first_cycle:
            if schedule_idx == -1:
                assert self.range == Range.START, f"the range should be 'start' for the first cycle. range: {self.range}"
                new_action = ActionWithLog(self.step, global_config.comm.pp_rank, self.microbatch, self.stage)
                schedule_idx = self.rank_schedule.add_action(new_action)
            elif schedule_idx == 0: # start of a new cycle. set is_first_cycle to False
                assert len(self.rank_schedule) <= global_config.parallelism.microbatches * global_config.parallelism.stages_per_rank * (4 if global_config.parallelism.bwd_separated else 2), f"the length of rank_schedule should be less than {global_config.parallelism.microbatches * global_config.parallelism.stages_per_rank * (4 if global_config.parallelism.bwd_separated else 2)}. rank_schedule: {len(self.rank_schedule)}"
                self.is_first_cycle = False
            
        self.step_cnt += (schedule_idx == 0) # count the step if it is the first action in the schedule

        
        # record the time event 
        new_event = torch.cuda.Event(enable_timing=True)
        new_event.record()
        self.cuda_timer_schedule[range].append(new_event)
        if schedule_idx == 0 and range == Range.START: # start of a new batch
            self.cuda_timer_batch[Range.START].append(new_event)
        elif schedule_idx == len(self.rank_schedule)-1 and range == Range.END: # end of a batch
            self.cuda_timer_batch[Range.END].append(new_event)

        # log the time duration 
        if self.step_cnt % self.log_freq == 0 \
            and schedule_idx == len(self.rank_schedule)-1 and range == Range.END:
            self.log_duration(schedule_idx, range)
        return self
    

    def log_duration(self, schedule_idx: int, range: Range) -> None:
        '''log the duration of the actions in the schedule, based on the monitored time events.'''
        if self.is_first_cycle:
            return
        # wait for the last event ends in GPU
        self.cuda_timer_schedule[Range.END][-1].synchronize()
        self.cuda_timer_batch[Range.END][-1].synchronize()
            
        assert len(self.cuda_timer_batch[Range.START]) == len(self.cuda_timer_batch[Range.END]), f"the length of batch start and end should be the same. start:{len(self.cuda_timer_batch[Range.START])} != end:{len(self.cuda_timer_batch[Range.END])}"
        assert len(self.cuda_timer_schedule[Range.START]) == len(self.cuda_timer_schedule[Range.END]), f"the length of schedule start and end should be the same. start:{len(self.cuda_timer_schedule[Range.START])} != end:{len(self.cuda_timer_schedule[Range.END])}"
        assert len(self.cuda_timer_schedule[Range.START]) % len(self.rank_schedule) == 0, \
            f"the length of schedule start should be a multiple of the rank schedule. start:{len(self.cuda_timer_schedule[Range.START])}, rank_schedule: {len(self.rank_schedule)}" 

        # log the time duration
        len_log = len(self.cuda_timer_batch[Range.START]) // len(self.rank_schedule)
        self.cuda_timer_schedule[Range.START] = np.array(self.cuda_timer_schedule[Range.START]).reshape(len_log, len(self.rank_schedule))            
        self.cuda_timer_schedule[Range.END] = np.array(self.cuda_timer_schedule[Range.END]).reshape(len_log, len(self.rank_schedule))
        for i, action in enumerate(self.rank_schedule):
            action.add_log_time([start.elapsed_time(end) for (start, end) in zip(self.cuda_timer_schedule[Range.START][:, i], self.cuda_timer_schedule[Range.END][:, i])])
        self.log_batch_time.extend([start.elapsed_time(end) for (start, end) in zip(self.cuda_timer_batch[Range.START], self.cuda_timer_batch[Range.END])])

        # reset the timer
        self.timer_reset()
        
        # print the average time for every {global_config.training.log_freq} steps (= {global_config.training.log_freq} batches)
        if self.step_cnt % ((global_config.training.log_freq//self.log_freq)*self.log_freq) == 1:
            emptiness = {type: len(stage_dict) == 0 for type, stage_dict in self.actions_list.items()} 
            avg_time = {type: np.mean([a.log_time for a_list in stage_dict.values() for a in a_list]) if not empty else 0 for ((type, stage_dict), empty) in zip(self.actions_list.items(), emptiness.values())}
            avg_batch_time = np.mean(self.log_batch_time)
            cnt_per_batch = {type: sum([a.get_log_time_len for a_list in stage_dict.values() for a in a_list]) // len(self.log_batch_time) for (type, stage_dict) in self.actions_list.items()}
            gpu_bubble_ratio = 1 - (sum([avg_time[type]*cnt_per_batch[type] for type in avg_time.keys()]) / avg_batch_time)
            log_str = f"Avg. fwd time: {avg_time[ActionType.FORWARD]:.4f} / "
            if global_config.parallelism.bwd_separated:
                if not emptiness[ActionType.FULL_BACKWARD]:
                    log_str += f"Avg. full-bwd time: {avg_time[ActionType.FULL_BACKWARD]:.4f} (Avg. bwd-weight time: {avg_time[ActionType.BACKWARD_WEIGHT]:.4f} / Avg. bwd-input time: {avg_time[ActionType.BACKWARD_INPUT]:.4f}) / "
                else:
                    log_str += f"Avg. bwd-weight time: {avg_time[ActionType.BACKWARD_WEIGHT]:.4f} / Avg. bwd-input time: {avg_time[ActionType.BACKWARD_INPUT]:.4f} / "
            else:
                log_str += f"Avg. bwd time: {avg_time[ActionType.FULL_BACKWARD]:.4f} / "
            log_str += f"Avg. batch time: {avg_batch_time:.4f} (ms) / GPU bubble ratio: {gpu_bubble_ratio*100:.2f}%"
            log_time(log_str)
        return
    
    # context manager
    def __enter__(self):
        '''enter the context'''
        assert self.disabled or self.range == Range.START, "the range should be 'start'"
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        '''exit the context'''
        # torch.cuda.synchronize() # wait for the GPU to finish
        # if not self.disabled and self.action_dict is not None and (self.step, global_config.comm.pp_rank, self.microbatch, self.stage) in self.action_dict:
        #     self.action_dict[(self.step, global_config.comm.pp_rank, self.microbatch, self.stage)].unfreeze()
        if not self.disabled:
            self(self.microbatch, self.stage, self.step, Range.END)
        return

    def forward(self, microbatch:int=-1, stage:int=None, postfix:str='')->'PipelineLog':
        return self(microbatch, stage, ActionType.FORWARD, Range.START, postfix=postfix) 
    def backward(self, microbatch:int=-1, stage:int=None, postfix:str='')->'PipelineLog':
        if not self.disabled and self.action_dict is not None:
            if (ActionType.FULL_BACKWARD, global_config.comm.pp_rank, microbatch, stage) in self.action_dict:
                self.action_dict[(ActionType.FULL_BACKWARD, global_config.comm.pp_rank, microbatch, stage)].freeze()
        return self(microbatch, stage, ActionType.FULL_BACKWARD, Range.START, postfix=postfix) 
    def backward_input(self, microbatch:int=-1, stage:int=None, postfix:str='')->'PipelineLog':
        if not self.disabled and self.action_dict is not None:
            if (ActionType.BACKWARD_WEIGHT, global_config.comm.pp_rank, microbatch, stage) in self.action_dict:
                self.action_dict[(ActionType.BACKWARD_WEIGHT, global_config.comm.pp_rank, microbatch, stage)].freeze()
        return self(microbatch, stage, ActionType.BACKWARD_INPUT, Range.START, postfix=postfix) 
    def backward_weight(self, microbatch:int=-1, stage:int=None, postfix:str='')->'PipelineLog':
        if not self.disabled and self.action_dict is not None:
            if (ActionType.BACKWARD_WEIGHT, global_config.comm.pp_rank, microbatch, stage) in self.action_dict:
                self.action_dict[(ActionType.BACKWARD_WEIGHT, global_config.comm.pp_rank, microbatch, stage)].freeze()
        return self(microbatch, stage, ActionType.BACKWARD_WEIGHT, Range.START, postfix=postfix) 
    def sync(self, microbatch:int=-1, stage:int=None, postfix:str='')->'PipelineLog':
        return self(microbatch, stage, ActionType.SYNC, Range.START, postfix=postfix) 


pipeline_logger : PipelineLogger = PipelineLogger()