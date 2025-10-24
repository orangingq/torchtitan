
from enum import Enum
from typing import List
import numpy as np
import torch
from torchtitan.tools.logging import logger


class ActionType(Enum):
    NONE = -1
    BUBBLE = 0
    FORWARD = 1
    BACKWARD_INPUT = 2
    BACKWARD_WEIGHT = 3
    START = 4
    SYNC = 5
    SEND_F = 6
    RECV_F = 7
    SEND_B = 8
    RECV_B = 9
    FULL_BACKWARD = 10
    FINISH = 11
    def __str__(self):
        return self.name
    def __repr__(self):
        return self.name
    def __int__(self):
        return self.value
    def __eq__(self, other):
        if isinstance(other, ActionType):
            return self.value == other.value
        elif isinstance(other, int):
            return self.value == other
        elif isinstance(other, str):
            return self.name.lower() == other.lower()
        else:
            return False
    def __hash__(self):
        return hash(self.value)

class Action:
    '''Action class for the pipeline'''
    def __init__(self, type:ActionType, rank:int, microbatch:int=None, stage:int=None):
        self.type:ActionType = type if isinstance(type, ActionType) else ActionType(int(type))
        self.rank:int = int(rank)
        self.microbatch:int = int(microbatch)
        self.stage = int(stage) if stage is not None else rank
    def __str__(self):
        return f"Action(type={self.type.name}, rank={self.rank}, microbatch={self.microbatch}, stage={self.stage})"
    def __repr__(self):
        return f"Action(type={self.type.name}, rank={self.rank}, microbatch={self.microbatch}, stage={self.stage})"
    def __eq__(self, other):
        if isinstance(other, Action):
            return self.type == other.type and self.rank == other.rank and self.microbatch == other.microbatch and self.stage == other.stage
        else:
            return False
    def to_tensor(self)-> torch.Tensor:
        return torch.tensor([int(self.type), self.rank, self.microbatch, self.stage])


class ActionWithLog(Action):
    '''Action class for the pipeline with time log'''
    def __init__(self, type:ActionType, rank:int, microbatch:int=None, stage:int=None, log_time:list=[]):
        super().__init__(type, rank, microbatch, stage)
        self.log_time = log_time.copy()

    def add_log_time(self, start_batch_idx:int, time:float|List[float]):
        '''add the log time starting from the given batch index'''
        if len(self.log_time) < start_batch_idx:
            logger.warning(f"Batch index {start_batch_idx} is larger than log time length {len(self.log_time)}. Filling with zeros.")
            self.log_time.extend([0.0] * (start_batch_idx - len(self.log_time)))
        elif len(self.log_time) > start_batch_idx:
            raise ValueError(f"Already have log time for batch index {start_batch_idx}.")
        
        if isinstance(time, list):
            self.log_time.extend(time.copy())
        else:
            self.log_time.append(time)
        return
    
    @property
    def get_log_time_len(self):
        '''get the log time length'''
        return len(self.log_time)
    @property
    def get_log_time_mean(self):
        '''get the log time mean'''
        return np.mean(self.log_time) if self.get_log_time_len > 0 else 0
    @property
    def get_log_time_median(self):
        '''get the log time median'''
        return np.median(self.log_time) if self.get_log_time_len > 0 else 0
    def __str__(self):
        return f"ActionWithLog(type={self.type.name}, rank={self.rank}, microbatch={self.microbatch}, stage={self.stage}, log_time[{self.get_log_time_len}]={self.get_log_time_mean:.4f})"
    def __repr__(self):
        return f"ActionWithLog(type={self.type.name}, rank={self.rank}, microbatch={self.microbatch}, stage={self.stage}, log_time[{self.get_log_time_len}]={self.get_log_time_mean:.4f})"
    def __eq__(self, other):
        if issubclass(type(other), Action):
            return self.type == other.type and self.rank == other.rank and self.microbatch == other.microbatch and self.stage == other.stage
        else:
            return False
    def to_tensor(self, with_log_time:bool=False, with_median:bool=False, with_mean:bool=False, log_window=None)->torch.Tensor:
        if not with_log_time and not with_median and not with_mean:
            return super().to_tensor()
        
        list_repr = super().to_tensor().tolist() # convert to list
        if with_median:
            list_repr.append(self.get_log_time_median if log_window is None else np.median(self.log_time[-log_window:]))
        if with_mean:
            list_repr.append(self.get_log_time_mean if log_window is None else np.mean(self.log_time[-log_window:]))
        # convert to tensor
        return torch.tensor(list_repr, dtype=torch.float16)



class ActionWithTime(Action):
    def __init__(self, type:ActionType, rank:int, microbatch:int=None, stage:int=None, duration:float=0):
        super().__init__(type, rank, microbatch, stage)
        # time setting
        self._start_time : float = 0 # starting time of this action block relative to the start time of this batch. # yet to be assigned
        self._duration = duration # set the duration of the action explictly

        # scheduled flag
        self.scheduled_flag = False # whether the action is scheduled or not
        self.prev_actions: list[Action] = [] # previous actions in the pipeline, used for scheduling
        self.next_actions: list[Action] = [] # next actions in the pipeline, used for scheduling      

    @property
    def start_time(self):
        '''Return the start time of the action.'''
        return self._start_time

    @start_time.setter
    def start_time(self, start_time:float):
        '''Set the start time and the end time of the action.'''
        self._start_time = max(0, start_time)
    
    @property
    def duration(self):
        '''Return the duration of the action.'''
        return self._duration
    
    @duration.setter
    def duration(self, duration:float):
        '''Set the duration of the action.'''
        self._duration = max(0, duration)
        return

    @property
    def end_time(self):
        '''Return the end time of the action.'''
        return self.start_time + self.duration
    
    def schedule(self):
        '''Schedule the action.'''
        self.scheduled_flag = True
    
    def __str__(self):
        return f"ActionWithTime(type={self.type.name}, rank={self.rank}, microbatch={self.microbatch}, stage={self.stage}, scheduled={self.scheduled_flag})"
    def __repr__(self):
        return f"ActionWithTime(type={self.type.name}, rank={self.rank}, microbatch={self.microbatch}, stage={self.stage}, scheduled={self.scheduled_flag})"



class ActionWithFreezing(ActionWithTime):
    def __init__(self, type:ActionType, rank:int, microbatch:int=None, stage:int=None, max_duration:float=0.0, min_duration:float=None):
        '''Action class for the pipeline with freezing capability.'''
        super().__init__(type, rank, microbatch, stage, max_duration)
        self._module = None # the module that this action is associated with and the unit of freezing # yet to be assigned
        self.num_params : int = None # number of parameters in the module # yet to be assigned

        # time setting
        self._max_duration : float = max_duration # maximum duration of this action, default is 0.0
        self._min_duration : float = max_duration # minimum duration of this action, default is max_duration
        if self.type == ActionType.BACKWARD_WEIGHT:
            self._min_duration = self._max_duration / 100 if min_duration is None else min_duration # initialize the minimum duration to 1% of the maximum duration
        elif self.type == ActionType.FULL_BACKWARD:
            self._min_duration = 0.5 * self._max_duration if min_duration is None else min_duration # initialize the minimum duration to half of the maximum duration
        assert self._min_duration <= self._max_duration, "Minimum duration must be less than or equal to maximum duration."

        # freezed flag
        self._freeze_flag : bool = False # whether the action can start freezing or not.
        self._expected_freeze_ratio : float = 0.0 # expected freeze ratio
        self.progressive_freezing : float = 0 # the rate of progressive freezing, default is 0. actual_fr = expected_fr * progressive_freezing (will grow to 1 during the training)
        self.monitored_points = [] # list of (afr, time_duration) pairs, monitored during the progressive freezing phase 

        self.freezing_list = None # list of freezing actions

        self.freeze_ratio_history = [] # frozen ratio history per stage
        self.paramwise_frozen_count = {} # [frozen, total] count for each layer in each stage

        # cache
        self.freeze_cache = {} # cache for requires_grad_ state of the parameters, key is the name of the parameter, value is requires_grad
        return
    
    @property
    def freezable(self):
        '''Whether this action block type is freezable or not regardless of the freeze flag or freeze ratio.'''
        return self.type in [ActionType.BACKWARD_WEIGHT, ActionType.FULL_BACKWARD]

    @property
    def min_duration(self):
        '''Return the minimum duration of the action.'''
        return self._min_duration
        
    
    @min_duration.setter
    def min_duration(self, min_duration:float):
        '''Set the minimum duration of the action.'''
        if self.freezable:
            self._min_duration = min(max(0, min_duration), self.max_duration)
        return

    @property
    def max_duration(self):
        '''Return the maximum duration of the action.'''
        return self._max_duration

    @max_duration.setter
    def max_duration(self, max_duration:float):
        '''Set the max duration of the action.'''
        self._max_duration = max(0, max_duration)
        return
    
    @property
    def duration(self, actual=False)-> float:
        '''
        Return the duration of the action.
        If actual is True, return the actual duration considering the freeze_flag and progressive freezing phase.
        If actual is False, return the expected duration.
        '''
        if actual:
            return self.min_duration + (self.max_duration - self.min_duration) * (1 - self.actual_freeze_ratio)
        return self.min_duration + (self.max_duration - self.min_duration) * (1 - self.expected_freeze_ratio)

    @duration.setter
    def duration(self, duration:float):
        '''Set the duration of the action, and corresponding expected freeze ratio.'''
        duration = max(min(duration, self.max_duration), self.min_duration) # clamp the duration between min and max duration
        assert self.min_duration <= duration <= self.max_duration, "Duration must be between min and max duration."
        if self.max_duration == self.min_duration:
            self.expected_freeze_ratio = 0.0
        else:
            self.expected_freeze_ratio = 1 - (duration - self.min_duration) / (self.max_duration - self.min_duration)
    
    @property
    def expected_freeze_ratio(self)-> float:
        '''Return the expected freeze ratio.'''
        return self._expected_freeze_ratio

    @expected_freeze_ratio.setter
    def expected_freeze_ratio(self, ratio:float):
        '''Set the expected freeze ratio.'''
        if self.freezable:
            self._expected_freeze_ratio = min(1, max(0, ratio)) # clamp the ratio between 0 and 1
        else:
            self._expected_freeze_ratio = 0.0

    @property
    def actual_freeze_ratio(self)-> float:
        '''Return the actual freeze ratio considering progressive freezing phase.'''
        if self.freeze_flag is False:
            return 0.0
        return self.expected_freeze_ratio * min(1, max(0, self.progressive_freezing))
    
    @property
    def module(self):
        '''Return the module that this action is associated with.'''
        return self._module

    @module.setter
    def module(self, module):
        '''Set the module that this action is associated with.'''
        self._module = module
        self.num_params : int = len(list(self._module.parameters()))
    
    @property
    def freeze_flag(self):
        '''Return the freeze flag.'''
        return self._freeze_flag

    @freeze_flag.setter
    def freeze_flag(self, freeze_flag:bool):
        '''
        Set the freeze flag.
        Freeze flag is set to True only if the action type is BACKWARD_WEIGHT or FULL_BACKWARD, and the module and num_params are set.
        '''
        freeze_flag = freeze_flag and self.freezable \
                    and (self.module is not None) and (self.num_params is not None)  
        if freeze_flag and len(self.paramwise_frozen_count) == 0:
            self.paramwise_frozen_count = {name: [0, 0] for name, _ in self.module.named_parameters()} # reset the paramwise frozen count
        self._freeze_flag = freeze_flag

    def freeze(self, start_batch_idx:int=None):
        '''Freeze the module. will be called before the backward pass.'''
        if not self.freeze_flag: # only freeze when the freeze flag is set.
            return
        
        max_p = 1 if self.stage == 0 else 0.995 # For the front layers, allow full parameter freezing
        if self.freezing_list is None:
            if self.actual_freeze_ratio <= 0:
                freezing_list = [False] * self.num_params
            else:
                expected = self.num_params * min(max_p, self.actual_freeze_ratio)
                lower = int(expected)
                p = expected - lower  # fractional part
                actual_num_freeze = lower + (1 if torch.rand(()) < p else 0)
                if self.stage == 0: # front layers more likely to freeze
                    weights = torch.linspace(1.0, 0.1, steps=self.num_params)
                    idx = torch.multinomial(weights, actual_num_freeze, replacement=False)
                else:
                    idx = torch.randperm(self.num_params)[:actual_num_freeze]
                freezing_list = torch.zeros(self.num_params, dtype=torch.bool)
                freezing_list[idx] = True
                freezing_list = freezing_list.tolist()
        else:
            freezing_list = self.freezing_list  

        for idx, (name, param) in enumerate(self.module.named_parameters()):
            self.freeze_cache[name] = param.requires_grad # cache the requires_grad state of the parameter
            param.requires_grad_(not freezing_list[idx]) # freeze the parameters by setting requires_grad to False

            self.paramwise_frozen_count[name][0] += int(param.requires_grad)
            self.paramwise_frozen_count[name][1] += 1
    
        # append the actual frozen ratio to the freeze ratio history
        if start_batch_idx is not None:
            if len(self.freeze_ratio_history) < start_batch_idx:
                logger.warning(f"Batch index {start_batch_idx} is larger than freeze ratio history length {len(self.freeze_ratio_history)}. Filling with zeros.")
                self.freeze_ratio_history.extend([0.0] * (start_batch_idx - len(self.freeze_ratio_history)))
            elif len(self.freeze_ratio_history) > start_batch_idx:
                raise ValueError(f"Already have freeze ratio for batch index {start_batch_idx}.")
        self.freeze_ratio_history.append(float(np.mean(freezing_list)))
        return
    
    def unfreeze(self):
        '''Unfreeze the module.'''
        if not self.freeze_flag:
            return
        
        for name, param in self.module.named_parameters():
            param.requires_grad_(self.freeze_cache[name]) # unfreeze all parameters
        return
    
    def to_tensor(self)-> torch.Tensor:
        return torch.tensor([int(self.type), self.rank, self.microbatch, self.stage, self.max_duration])

    def __str__(self):
        return f"ActionWithFreezing(type={self.type.name}, rank={self.rank}, microbatch={self.microbatch}, stage={self.stage}, freezing={self.expected_freeze_ratio})"
    def __repr__(self):
        return f"ActionWithFreezing(type={self.type.name}, rank={self.rank}, microbatch={self.microbatch}, stage={self.stage}, freezing={self.expected_freeze_ratio})"

