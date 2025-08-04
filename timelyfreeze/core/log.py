from enum import Enum
from typing import Dict, List, Union
import numpy as np
import torch
import timeit

from .util import log_time
from .action import Action, ActionType, ActionWithLog, ActionWithFreezing
from .config import global_config
from .nvtx import forward_nvtx, backward_nvtx, backward_input_nvtx, mark_nvtx, grad_sync_nvtx, comm_nvtx

__all__ = ['pipeline_log', 'PipelineLog']

class Range:
    START:str = 'start'
    END:str = 'end'
    NONE:None = None
    ALL = [START, END, NONE]

class ActionPhase:
    START = [ActionType.START]
    FORWARD = [ActionType.RECV_F, ActionType.FORWARD, ActionType.SEND_F]
    BACKWARD = [ActionType.RECV_B, ActionType.FULL_BACKWARD, ActionType.SEND_B]
    END = [ActionType.SYNC, ActionType.FINISH]

class LogType(Enum):
    '''log type. rank or batch'''
    BATCH = 0
    RANK = 1
    def __str__(self):
        return self.name
    def __repr__(self):
        return self.name
    def __int__(self):
        return self.value
    def __eq__(self, other):
        if isinstance(other, LogType):
            return self.value == other.value
        elif isinstance(other, int):
            return self.value == other
        elif isinstance(other, str):
            return self.name.lower() == other.lower()
        else:
            return False


class PipelineLog:
    '''log the pipeline step.'''
    def __init__(self, type:Union[LogType, str]=LogType.BATCH, log:bool=True) -> None:
        assert type in [LogType.BATCH, LogType.RANK], f"type should be {LogType.BATCH} or {LogType.RANK}"
        self.type = type
        self.stage = global_config.comm.pp_rank
        self.print_log = log
        self.nvtx_log = True
        self.microbatch = None
        self.step = None
        self.range = None
        self.step_name = None
        self.last_log_time = timeit.default_timer()
        self.step_cnt = -1
        self.set_format() # set the print format
        self.disabled = False # temporary disable flag. cannot call the __call__ function if set to True

        # nvtx functions
        self.mark_nvtx = mark_nvtx
        self.forward_nvtx = forward_nvtx
        self.backward_nvtx = backward_nvtx
        self.backward_input_nvtx = backward_input_nvtx
        self.grad_sync_nvtx = grad_sync_nvtx
        self.comm_nvtx = comm_nvtx

        # cuda timer
        self.cuda_timer_batch = {Range.START: [], Range.END: []}
        self.cuda_timer_schedule = {Range.START: [], Range.END: []}
        # log time
        self.log_batch_time = []
        self.actions_list = {
            ActionType.FORWARD: {}, 
            ActionType.FULL_BACKWARD: {},
            ActionType.BACKWARD_INPUT: {},
            ActionType.BACKWARD_WEIGHT: {},
        }
        # log the schedule order
        self.log_schedule_flag = True
        self.log_schedule:List[ActionWithLog] = []
        self.pipeline_schedule:List[List[ActionWithFreezing]] = None # the pipeline schedule, which is a list of list of ActionWithFreezing
        self.action_dict: Dict[tuple, ActionWithFreezing] = None # the action dict, which is a dict of ActionWithFreezing with key (step, pp_rank, microbatch, stage)
        return

    def initialize(self, log:bool=False):
        '''initialize the log'''
        if global_config.parallelism.bwd_separated:
            ActionPhase.BACKWARD = [ActionType.RECV_B, ActionType.FULL_BACKWARD, ActionType.BACKWARD_INPUT, ActionType.BACKWARD_WEIGHT, ActionType.SEND_B]
        self.set_log(log)
        self.microbatch = -1
        self.step = None
        self.range = None
        self.step_name = None
        self.last_log_time = timeit.default_timer()
        self.step_cnt = 0
        self.set_format()
        self.timer_reset()
        self.log_batch_time = []
        self.actions_list = { # ActionType -> {stage -> [time]}
            ActionType.FORWARD: {}, 
            ActionType.FULL_BACKWARD: {},
            ActionType.BACKWARD_INPUT: {},
            ActionType.BACKWARD_WEIGHT: {},
        }
        return
    
    def timer_reset(self):
        '''reset the timer'''
        self.cuda_timer_batch = {Range.START: [], Range.END: []}
        self.cuda_timer_schedule = {Range.START: [], Range.END: []}
        return

    def get_log(self):
        '''get the log'''
        return self.print_log
    
    def set_log(self, log:bool):
        '''set the log'''
        self.print_log = log
        return
    
    def disable(self):
        '''disable the log'''
        self.disabled = True
        return

    def enable(self):
        '''enable the log'''
        self.disabled = False
        return

    def __call__(self, microbatch:int=-1, stage:int=None, step:ActionType='', range:Range=Range.NONE, postfix:str='', timestamp=None)->'PipelineLog':
        '''log the pipeline step.
        Args:
            microbatch: int, the microbatch index
            step: Union[ActionType, str], the step name or ActionType number
            range: ['start', 'end', 'None']. If 'start', push the nvtx range. If 'end', pop the nvtx range. If 'None', mark the time point. 
            postfix: str, the postfix of the step
            timestamp: float, the timestamp of the step
        '''
        if self.disabled or not (self.nvtx_log or self.print_log):
            return self
        elif self.step_cnt >= 5:
            self.print_log = False
        
        assert step in ActionPhase.START + ActionPhase.FORWARD + ActionPhase.BACKWARD + ActionPhase.END, f"step {step} should be in {ActionPhase.START + ActionPhase.FORWARD + ActionPhase.BACKWARD + ActionPhase.END}"
        assert range in Range.ALL, "range should be 'start', 'end' or 'None'"
        
        if microbatch != -1:
            self.microbatch = microbatch
        self.stage = stage if stage is not None else global_config.comm.pp_rank
        self.step = step
        self.range = range
        if step in ActionPhase.START:
            self.step_cnt += 1 # count the step
        
        if self.nvtx_log:
            self._mark_nvtx(range=range)
            self.log_timer()
        if not self.print_log:
            return self
        
        self._set_step_name()
        if step in ActionPhase.START: 
            self.set_format()
            if global_config.comm.is_first_stage:
                log_time(f" {'|'.join(self.start_format)}  - {step:9} {postfix}", rank=(self.type!=LogType.RANK), timestamp=timestamp)
            return self
        elif step == ActionType.FINISH:
            if global_config.comm.is_first_stage:
                log_time(f" {'#'.join(self.end_format)}  - {step:9} {postfix}", rank=(self.type!=LogType.RANK), timestamp=timestamp, end='\n\n')
            return self
        elif step == ActionType.SYNC:
            if range == Range.START:
                log_time(f"[ {' | '.join(self.format)} ] - {step:9} {postfix}", rank=(self.type!=LogType.RANK), timestamp=timestamp)
            if range == Range.END and global_config.comm.is_last_stage:
                log_time(f" {'#'.join(self.end_format)}  - {step:9} {postfix}", rank=(self.type!=LogType.RANK), timestamp=timestamp)
            return self
        elif not step in ActionPhase.FORWARD + ActionPhase.BACKWARD + ActionPhase.END:
            return self

        # add time duration
        now_time = timeit.default_timer()
        if range == Range.END or step == ActionType.FINISH:
            postfix = f"{now_time - self.last_log_time:.4f}s " + postfix
        self.last_log_time = now_time

        # log the step
        format_copy = self.format.copy()
        if self.type == LogType.RANK:
            format_copy[global_config.comm.pp_rank] = self.step_name
        else:
            format_copy[microbatch] = f"{self.step_name:{len(self.format[0])}d}"
        log_time(f"[ {' | '.join(format_copy)} ] - {step:9} {postfix}", rank=(self.type!=LogType.RANK), timestamp=timestamp)
        return self
    
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
        self(self.microbatch, self.stage, self.step, Range.END)
        return

    def fwd_recv(self, microbatch:int=-1, stage:int=None, postfix:str='')->'PipelineLog':
        log_time(f"{self.microbatch}F{stage}R")
        return self(microbatch, stage, ActionType.RECV_F, Range.START, postfix=postfix) 
    def forward(self, microbatch:int=-1, stage:int=None, postfix:str='')->'PipelineLog':
        return self(microbatch, stage, ActionType.FORWARD, Range.START, postfix=postfix) 
    def fwd_send(self, microbatch:int=-1, stage:int=None, postfix:str='')->'PipelineLog':
        log_time(f"{self.microbatch}F{stage}S")
        return self(microbatch, stage, ActionType.SEND_F, Range.START, postfix=postfix) 
    def bwd_recv(self, microbatch:int=-1, stage:int=None, postfix:str='')->'PipelineLog':
        return self(microbatch, stage, ActionType.RECV_B, Range.START, postfix=postfix) 
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
    def bwd_send(self, microbatch:int=-1, stage:int=None, postfix:str='')->'PipelineLog':
        return self(microbatch, stage, ActionType.SEND_B, Range.START, postfix=postfix) 
    def sync(self, microbatch:int=-1, stage:int=None, postfix:str='')->'PipelineLog':
        return self(microbatch, stage, ActionType.SYNC, Range.START, postfix=postfix) 

    def log_timer(self):
        '''log the timer'''
        if not self.step in [ActionType.FORWARD, ActionType.FULL_BACKWARD, ActionType.BACKWARD_INPUT, ActionType.BACKWARD_WEIGHT]:
            return
        if not self.range in [Range.START, Range.END]:
            return

        # Process to identify pipeline schedule cycle (for the first cycle)
        if self.log_schedule_flag:
            if not self.range == Range.START:
                return
            new_action = ActionWithLog(self.step, global_config.comm.pp_rank, self.microbatch, self.stage)
            if len(self.log_schedule) > 0 and self.log_schedule[0] == new_action: # start of a new pipeline cycle -> end the log_schedule
                self.log_schedule_flag = False
                assert len(self.log_schedule) <= global_config.parallelism.microbatches * global_config.parallelism.stages_per_rank * (4 if global_config.parallelism.bwd_separated else 2), f"the length of log_schedule should be less than {global_config.parallelism.microbatches * global_config.parallelism.stages_per_rank * (4 if global_config.parallelism.bwd_separated else 2)}. log_schedule: {len(self.log_schedule)}"
            else:
                self.log_schedule.append(new_action)
                if self.stage in self.actions_list[self.step].keys():
                    self.actions_list[self.step][self.stage].append(new_action)
                else:
                    self.actions_list[self.step][self.stage] = [new_action]
                return # pass the first cycle
        
        # record the time for the start and end of each step
        new_event = torch.cuda.Event(enable_timing=True)
        new_event.record()

        # record the time for the start and end of each step
        assert Action(self.step, global_config.comm.pp_rank, self.microbatch, self.stage) in self.log_schedule, f"the action ({self.step}, {global_config.comm.pp_rank}, {self.microbatch}, {self.stage}) should be in the log_schedule. log_schedule: {self.log_schedule}"
        self.cuda_timer_schedule[self.range].append(new_event)

        # record the time for the start of each batch
        start_of_batch = self.step == ActionType.FORWARD and self.range == Range.START and self.microbatch == 0 and self.stage == global_config.comm.pp_rank
        end_of_batch = self.step in [ActionType.FULL_BACKWARD, ActionType.BACKWARD_WEIGHT] and self.range == Range.END and self.microbatch == global_config.parallelism.microbatches-1 and self.stage == global_config.comm.pp_rank
        if start_of_batch or end_of_batch:
            self.cuda_timer_batch[self.range].append(new_event)

        # log the time duration of forward and backward, for every n=10 batches
        n = 10
        end_of_nth_batch = end_of_batch and self.step_cnt % n == 1 and self.step_cnt > 1
        if end_of_nth_batch:
            assert len(self.cuda_timer_batch[Range.START]) == n, f"the length of batch start should be {n}. start:{len(self.cuda_timer_batch[Range.START])}, end:{len(self.cuda_timer_batch[Range.END])}"
            assert len(self.cuda_timer_schedule[Range.START]) == n * len(self.log_schedule), f"the length of fwd start should be {n} * {len(self.log_schedule)}(=len(self.log_schedule)). start:{len(self.cuda_timer_schedule[Range.START])}, end:{len(self.cuda_timer_schedule[Range.END])}"
            assert self.step_cnt % n == 1, f"the step count should be a multiple of {n}. step_cnt: {self.step_cnt}"

            # wait for the last event ends in GPU
            self.cuda_timer_schedule[Range.END][-1].synchronize()
            self.cuda_timer_batch[Range.END][-1].synchronize()
                
            assert len(self.cuda_timer_batch[Range.START]) == len(self.cuda_timer_batch[Range.END]), f"the length of batch start and end should be the same. start:{len(self.cuda_timer_batch[Range.START])} != end:{len(self.cuda_timer_batch[Range.END])}"
            assert len(self.cuda_timer_schedule[Range.START]) == len(self.cuda_timer_schedule[Range.END]), f"the length of schedule start and end should be the same. start:{len(self.cuda_timer_schedule[Range.START])} != end:{len(self.cuda_timer_schedule[Range.END])}"
            
            # log the time duration
            self.cuda_timer_schedule[Range.START] = np.array(self.cuda_timer_schedule[Range.START]).reshape(n, len(self.log_schedule))            
            self.cuda_timer_schedule[Range.END] = np.array(self.cuda_timer_schedule[Range.END]).reshape(n, len(self.log_schedule))
            for i, action in enumerate(self.log_schedule):
                action.add_log_time([start.elapsed_time(end) for (start, end) in zip(self.cuda_timer_schedule[Range.START][:, i], self.cuda_timer_schedule[Range.END][:, i])])
            self.log_batch_time.extend([start.elapsed_time(end) for (start, end) in zip(self.cuda_timer_batch[Range.START], self.cuda_timer_batch[Range.END])])

            # reset the timer
            self.timer_reset()
            # print the average time for every {global_config.training.log_freq} steps (= {global_config.training.log_freq} batches)
            if self.step_cnt % ((global_config.training.log_freq//n)*n) == 1:
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

    def _set_step_name(self):
        '''set the step name'''
        if self.type == LogType.RANK:
            if self.step in ActionPhase.START:
                self.step_name = '##'
            if self.step in ActionPhase.FORWARD:
                self.step_name = str(self.stage) + 'F' + str(self.microbatch)
            elif self.step in ActionPhase.BACKWARD:
                self.step_name = str(self.stage) + 'B' + str(self.microbatch)
            elif self.step in ActionPhase.END:
                self.step_name = str(self.stage) + 'E' + str(self.microbatch)
        else:
            if self.step in ActionPhase.START:
                self.step_name = '##'
            if self.step in ActionPhase.FORWARD:
                self.step_name = ActionPhase.FORWARD.index(self.step) + len(ActionPhase.FORWARD)*global_config.comm.pp_rank -1 # -1 is because the first rank has no 'fwd_recv'
            elif self.step in ActionPhase.BACKWARD:
                self.step_name = ActionPhase.BACKWARD.index(self.step) + (len(ActionPhase.FORWARD)* global_config.comm.pp - 2) + len(ActionPhase.BACKWARD)*(global_config.comm.pp-1 - global_config.comm.pp_rank) -1
                # -2 is because the first rank has no 'fwd_recv' and the last rank has no 'fwd_send'
                # -1 is because the last rank has no 'bwd_recv'
            elif self.step in ActionPhase.END:
                self.step_name = ActionPhase.END.index(self.step) + (len(ActionPhase.FORWARD)* global_config.comm.pp - 2) + (len(ActionPhase.BACKWARD)*global_config.comm.pp -2)
                # -2 in forward is because the first rank has no 'fwd_recv' and the last rank has no 'fwd_send'
                # -2 in backward is because the last rank has no 'bwd_recv' and the first rank has no 'bwd_send'
        return
        
    def _mark_nvtx(self, range):
        '''mark on the nvtx'''
        # set nvtx 
        if range == Range.NONE:
            self.mark_nvtx(self.step, self.step_cnt)
        elif self.step == ActionType.FORWARD:
            self.forward_nvtx(self.microbatch, start=(range==Range.START))
        elif self.step == ActionType.FULL_BACKWARD:
            self.backward_nvtx(self.microbatch, start=(range==Range.START))
        elif self.step == ActionType.BACKWARD_INPUT:
            self.backward_input_nvtx(self.microbatch, start=(range==Range.START))
        elif self.step == ActionType.BACKWARD_WEIGHT:
            self.backward_nvtx(self.microbatch, start=(range==Range.START))
        elif self.step == ActionType.SYNC:
            self.grad_sync_nvtx(self.microbatch, start=(range==Range.START))
        else:
            self.comm_nvtx(self.step, self.microbatch, start=(range==Range.START))
        return
    
    def set_format(self):
        '''set the format'''
        if self.type == LogType.RANK: # print per rank
            self.start_format = [f'GPU{k}'.center((len(str(global_config.parallelism.microbatches))+4)) for k in range(global_config.comm.pp)]
            self.format = ['-'*(len(str(global_config.parallelism.microbatches))+2)]*global_config.comm.pp
            self.end_format = ['#'*(len(str(global_config.parallelism.microbatches))+4)]*global_config.comm.pp
        else: # print per microbatch
            final_step = len(ActionPhase.FORWARD)* global_config.comm.pp + len(ActionPhase.BACKWARD)*global_config.comm.pp + len(ActionPhase.END)
            self.start_format = [f'MB{k}'.center((len(str(final_step))+2)) for k in range(global_config.parallelism.microbatches)]
            self.format = ['-'*len(str(final_step))]*global_config.parallelism.microbatches
            self.end_format = ['#'*(len(str(final_step))+2)]*global_config.parallelism.microbatches
        return
    
    
pipeline_log = PipelineLog(type=LogType.RANK, log=False)
