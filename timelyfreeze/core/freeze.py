

import copy
from typing import List
import numpy as np
from .action import ActionType, ActionWithFreezing, ActionWithLog, ActionWithTime
from .log import pipeline_log
from .schedule import adjust_freeze_ratio, gather_pipeline_schedule, set_freeze_ratio
from .config import global_config
from .util import log_time

class _Freezer:
    def __init__(self, model):
        self.model = model
        self.step_cnt = 0
        self.epoch_cnt = 0

        # frozen status
        self.stability_check_freq = 10 # check the stability every 50 steps (paper: 50)
        self.is_frozen = {name: False for name, _ in model.named_parameters()}
        self.freeze_ratio = 0.0
        self.freeze_ratio_history = [0] * (global_config.training.warmup_steps * global_config.training.num_batches // self.stability_check_freq) if global_config.freezing.freeze else []
        self.paramwise_frozen_count = {name: [0, 0] for name, _ in model.named_parameters()} # [frozen, total] count for each layer

        # early stopping
        self.early_stopping = False
        return

    def _step_count(self):
        '''Count the number of steps and epochs. 
        Call this function at the start of each forward pass.
        Starting from (epoch 0, step 1) for the first microbatch.
        '''
        if self.model.training: # only count the steps during training
            self.step_cnt += 1
            if self.step_cnt % global_config.training.num_batches == 1:
                self.epoch_cnt += 1
        return
    
    def freeze_update(self):
        self._step_count()
        if not global_config.freezing.freeze or self.epoch_cnt <= global_config.training.warmup_steps:
            return
        
        if self.step_cnt % self.stability_check_freq == 0:
            self._calculate_freezing_metric()
            self._threshold_update()
            self._freeze_update() 
            self._update_freeze_ratio()
            log_time(f"Freezing Ratio: {self.freeze_ratio:.2f}, Threshold: {self.threshold:.4e}")
        return

    def _calculate_freezing_metric(self):
        raise NotImplementedError("This function should be implemented in the derived class.")
        pass

    def _threshold_update(self):
        raise NotImplementedError("This function should be implemented in the derived class.")
        pass

    def _freeze_update(self):
        raise NotImplementedError("This function should be implemented in the derived class.")
        pass

    def _update_freeze_ratio(self):
        '''Update the frozen ratio based on the current freezing status.'''
        if self.step_cnt % self.stability_check_freq != 0:
            return

        for name, _ in self.model.named_parameters():
            self.paramwise_frozen_count[name][0] += self.is_frozen[name]
            self.paramwise_frozen_count[name][1] += 1
    
        self.freeze_ratio = sum(self.is_frozen.values()) / len(self.is_frozen)
        self.freeze_ratio_history.append(self.freeze_ratio)
        return

class _Freezer_v3_0714(_Freezer):
    def __init__(self, model):
        '''Version 2: Updated on July 14, 2025
        - Collaborate with ActionWithFreezing to freeze per microbatch block.
        '''
        self.model = model
        self.step_cnt = 0
        self.epoch_cnt = 0

        # frozen status
        self.stability_check_freq = 10 # check the stability every 50 steps (paper: 50)

        # Warmup Phase: do nothing
        self.warmup_phase = max(1, global_config.training.warmup_steps - 1) # first epoch is the warmup phase
        # Monitoring Phase: do not freeze the model, only analyze the time.
        self.monitoring_phase = self.warmup_phase + 1 # at least 1 epoch for monitoring
        self.is_monitoring = True # set to False at the first stability check freq after monitoring phase
        # Progressive Freezing Phase: gradually increase the freezing_params_num to the expected number.
        self.progressive_freezing_phase = (self.monitoring_phase + 2) if 'debug' in global_config.training.basename else (self.monitoring_phase + 5)
        self.log_epoch_cnt = 0

        self.freeze_ratio_history = {stage.stage_idx: [0] * (self.monitoring_phase * global_config.training.num_batches // self.stability_check_freq) if global_config.freezing.freeze else [] for stage in model} # frozen ratio history per stage
        self.paramwise_frozen_count = {stage.stage_idx: {name: [0, 0] for name, _ in stage.named_parameters()} for stage in model} # [frozen, total] count for each layer in each stage
        return

    def _step_count(self):
        '''Count the number of steps and epochs. 
        Call this function at the start of each forward pass.
        Starting from (epoch 1, step 1) for the first microbatch.
        '''
        if self.model.training: # only count the steps during training
            self.step_cnt += 1
            if self.step_cnt % global_config.training.num_batches == 1:
                self.epoch_cnt += 1
        return
    

    def freeze_update(self):
        self._step_count()
        if not global_config.freezing.freeze or self.epoch_cnt <= self.warmup_phase:
            return
        
        if self.step_cnt % self.stability_check_freq == 0:
            self._set_expected_freeze_ratio()
        
            if self.monitoring_phase < self.epoch_cnt:
                self._update_freeze_ratio()
            if self.log_epoch_cnt < self.epoch_cnt and self.monitoring_phase < self.epoch_cnt and self.epoch_cnt <= self.progressive_freezing_phase + 2:
                self.log_epoch_cnt = self.epoch_cnt
                log_time(f"Current/Expected Freeze Ratio per Block: {', '.join([f'[MB{action.microbatch}] {action.actual_freeze_ratio:.2f}/{action.expected_freeze_ratio:.2f}' for action in pipeline_log.pipeline_schedule[global_config.comm.pp_rank] if action.type in [ActionType.BACKWARD_WEIGHT, ActionType.FULL_BACKWARD]])}")
        return

    def _set_expected_freeze_ratio(self):
        raise NotImplementedError("This function should be implemented in the derived class.")


    def _update_freeze_ratio(self):
        '''Update the frozen ratio based on the current freezing status.'''
        if self.step_cnt % self.stability_check_freq != 0:
            return

        for stage in self.model:    
            for name, _ in stage.named_parameters():
                self.paramwise_frozen_count[stage.stage_idx][name][0] = sum([a.paramwise_frozen_count[name][0] for a in pipeline_log.pipeline_schedule[global_config.comm.pp_rank] if a.stage == stage.stage_idx and name in a.paramwise_frozen_count])
                self.paramwise_frozen_count[stage.stage_idx][name][1] = sum([a.paramwise_frozen_count[name][1] for a in pipeline_log.pipeline_schedule[global_config.comm.pp_rank] if a.stage == stage.stage_idx and name in a.paramwise_frozen_count])

            average_freeze_ratio = float(np.mean([a.actual_freeze_ratio for a in pipeline_log.pipeline_schedule[global_config.comm.pp_rank] if a.stage == stage.stage_idx and a.type in [ActionType.BACKWARD_WEIGHT, ActionType.FULL_BACKWARD]]))
            self.freeze_ratio_history[stage.stage_idx].append(average_freeze_ratio)
        return

def get_freezer_class_version(freezer:_Freezer)->int:
    '''Get the version of the freezer class.'''
    if isinstance(freezer, _Freezer):
        if issubclass(freezer.__class__, _Freezer_v3_0714):
            return 1
        else:
            return 0
    else:
        raise TypeError(f"Freezer should be an instance of _Freezer, but got {type(freezer)}.")

def get_freezer(model)->_Freezer:
    '''Get the freezer based on the metric type.'''
    if global_config.freezing.metric_type == 'fullrand6': # freeze per microbatch block - updated on July 14, 2025
        return FullyRandomFreezer_v6(model)
    elif global_config.freezing.metric_type == 'apf': # APF (absolute perturbation freezing)
        return APFFreezer(model)
    else:
        raise NotImplementedError(f"Metric Type [{global_config.freezing.metric_type}] is not supported.")


class FullyRandomFreezer_v6(_Freezer_v3_0714):
    def __init__(self, model):
        ''' Updated on July 14, 2025
        Set different expected freeze ratio per microbatch block.
        '''
        super().__init__(model)

        self.monitored_ub = False # True if monitored the upperbound of batch time. set to True at the first stability check freq after monitoring phase. 
        self.monitoring_lb = False # True if currently monitoring the lowerbound of batch time, i.e., freezing ratio = 1.0 for all actions.
        self.monitored_lb = False # True if monitored the lowerbound of batch time, i.e., freezing ratio = 1.0 for all actions.
        self.monitoring_lb_start = None # starting step of monitoring lowerbound which is used in second monitoring phase
        self.max_batch_time = None
        # self.freeze_adjust_freq = self.stability_check_freq * (1 + args.num_batches//self.stability_check_freq)
        return

    def freeze_update(self):
        self._step_count()
        if not global_config.freezing.freeze or self.epoch_cnt <= self.warmup_phase:
            return
        
        if self.step_cnt % self.stability_check_freq == 0:
            self._set_expected_freeze_ratio()
        
            if self.monitoring_phase < self.epoch_cnt:
                self._update_freeze_ratio()

            # log the current and expected freeze ratio per microbatch block
            if self.log_epoch_cnt < self.epoch_cnt and self.monitoring_phase < self.epoch_cnt and self.epoch_cnt <= self.progressive_freezing_phase + 2:
                self.log_epoch_cnt = self.epoch_cnt
                log_time(f"Current/Expected Freeze Ratio per Block: {', '.join([f'[MB{action.microbatch}] {action.actual_freeze_ratio:.2f}/{action.expected_freeze_ratio:.2f}' for action in pipeline_log.pipeline_schedule[global_config.comm.pp_rank] if action.type in [ActionType.BACKWARD_WEIGHT, ActionType.FULL_BACKWARD]])}")
        return

    def _set_expected_freeze_ratio(self):
        '''Set the expected freeze ratio based on the backward time.'''
        if self.epoch_cnt <= self.monitoring_phase:
            # during the monitoring phase, do not freeze the model
            self.monitored_ub = False
            return
        elif self.epoch_cnt < self.progressive_freezing_phase:
            if not self.monitored_ub: # first stability check freq after monitoring phase
                pipeline_schedule :List[List[ActionWithFreezing]] = set_freeze_ratio(gather_pipeline_schedule(pipeline_log.log_schedule))

                # Set the stage module for each action in the pipeline schedule
                stage_dict = {stage.stage_idx: stage for stage in self.model}
                for a in pipeline_schedule[global_config.comm.pp_rank]:
                    a.module = stage_dict[a.stage]
                    a.freeze_flag = True
                pipeline_log.pipeline_schedule = pipeline_schedule
                pipeline_log.action_dict = {(action.type, action.rank, action.microbatch, action.stage): action \
                                                    for action in pipeline_schedule[global_config.comm.pp_rank]}
                self.monitored_ub = True

            # during the warmup phase, gradually increase the progressive_freezing
            for a in pipeline_log.pipeline_schedule[global_config.comm.pp_rank]:
                a.progressive_freezing = (self.epoch_cnt-self.monitoring_phase)/(self.progressive_freezing_phase-self.monitoring_phase)
            return
    
        # during the last 10 steps of the progressive freezing phase, monitor the lowerbound of batch time and set min_duration of each action block
        elif self.epoch_cnt == self.progressive_freezing_phase: 
            if not self.monitoring_lb and not self.monitored_lb: # start lowerbound monitoring phase
                for a in pipeline_log.pipeline_schedule[global_config.comm.pp_rank]:
                    a.progressive_freezing = 1.0
                    if a.stage == global_config.parallelism.num_stages - 1: # last stage
                        a.expected_freeze_ratio = 1.0 - 1/a.num_params
                    else:
                        a.expected_freeze_ratio = 1.0
                self.monitoring_lb = True
                self.monitoring_lb_start = pipeline_log.step_cnt

            elif self.monitoring_lb and not self.monitored_lb: # end lowerbound monitoring phase
                if len(pipeline_log.log_schedule[0].log_time) <= self.monitoring_lb_start + 20 :
                    # not enough log time data for lowerbound monitoring
                    return
                # create lowerbound pipeline schedule
                log_schedule_lb :List[ActionWithLog] = copy.deepcopy(pipeline_log.log_schedule)
                for la, a, pa in zip(log_schedule_lb, pipeline_log.log_schedule, pipeline_log.pipeline_schedule[global_config.comm.pp_rank]):
                    if a.type in [ActionType.FULL_BACKWARD, ActionType.BACKWARD_WEIGHT]:
                        la.log_time = a.log_time[self.monitoring_lb_start:]
                        assert la.get_log_time_mean < pa.max_duration, f"Lowerbound log time {la.get_log_time_mean} is not less than action log time {pa.max_duration}."
                    else:
                        la.log_time = a.log_time[:self.monitoring_lb_start]
                pipeline_schedule_lb :List[List[ActionWithTime]] = gather_pipeline_schedule(log_schedule_lb)

                # set the min_duration of each action block based on the lowerbound log time
                for ar_lb, actions_per_rank in zip(pipeline_schedule_lb, pipeline_log.pipeline_schedule):
                    for a_lb, a in zip(ar_lb, actions_per_rank):
                        if a.type in [ActionType.FULL_BACKWARD, ActionType.BACKWARD_WEIGHT]:
                            a.min_duration = a_lb.duration
                
                pipeline_log.pipeline_schedule = set_freeze_ratio(pipeline_log.pipeline_schedule)

                self.monitored_lb = True
                self.monitoring_lb = False
            else:
                # after the warmup phase, freeze the model at the expected freeze ratio
                for a in pipeline_log.pipeline_schedule[global_config.comm.pp_rank]:
                    a.progressive_freezing = 1
                return
            return
        


class APFFreezer(_Freezer):
    '''
    ** Baseline Paper : [ICDCS'21] Communication-Efficient Federated Learning with Adaptive Parameter Freezing
        - https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9546506

    Update the freezing status based on the absolute perturbation freezing (APF).
    - Metric = |EMA[grad]| / EMA[|grad|]
    - Threshold = 0.05 (default)
    - Freeze the layer if Metric < Threshold in TCP style.
    '''
    def __init__(self, model):
        super().__init__(model)

        # freezing metrics
        self.stability_check_freq = 50 # frequency of stability check (paper: 50)
        self.alpha = 0.99 # parameter for exponential moving average (paper: 0.99)
        self.last_param = {name: None for name, _ in model.named_parameters()} # cumulative update
        self.ema = {name: 0.0 for name, _ in model.named_parameters()}
        self.ema_abs = {name: 0.0 for name, _ in model.named_parameters()}
        self.freezing_metric = {name: 0.0 for name, _ in model.named_parameters()} # effective perturbation metric. -> 0: oscillation (stable), -> 1: directional (unstable)

        # freeze update
        self.threshold = 0.05 # threshold on effective perturbation. (paper: 0.05)
        self.freezing_period = {name: 0 for name, _ in model.named_parameters()} # freeze the layer for a unit of stability_check_freq (paper: 50)
        self.frozen_due = {name: 0 for name, _ in model.named_parameters()}
        self.is_frozen = {name: False for name, _ in model.named_parameters()}
        return
    
    def _calculate_freezing_metric(self):
        '''Calculate the effective perturbation metric for parameters.'''
        for name, param in self.model.named_parameters():
            if self.is_frozen[name] or param.grad is None:
                continue
            
            curr_param = param.detach().clone().cpu()  # current parameter value
            if self.last_param[name] is None:
                grad, grad_abs = 0.0, 0.0
            else:
                grad = (curr_param - self.last_param[name]).mean() # param.grad.mean().item()
                grad_abs = (curr_param - self.last_param[name]).abs().mean() # param.grad.abs().mean().item()

            # calculate the effective perturbation metric
            self.ema[name] = self.alpha * self.ema[name] + (1 - self.alpha) * grad
            self.ema_abs[name] = self.alpha * self.ema_abs[name] + (1 - self.alpha) * grad_abs
            self.freezing_metric[name] = abs(self.ema[name]) / max(self.ema_abs[name], 1e-10)
            self.last_param[name] = curr_param  # reset the cumulative update after calculating the metric
        return

    def _freeze_update(self):
        '''Update the freezing status of the model parameters based on the threshold.'''
        for name, param in self.model.named_parameters():
            if not self.is_frozen[name]:
                # update the freezing period (update in TCP style)
                if self.freezing_metric[name] <= self.threshold:
                    self.freezing_period[name] += self.stability_check_freq
                else:
                    self.freezing_period[name] = self.freezing_period[name]/2
                    if self.freezing_period[name] < self.stability_check_freq:
                        self.freezing_period[name] = 0

                # update the freezing deadline
                self.frozen_due[name] = self.step_cnt + self.freezing_period[name]

            # update the freezing status
            self.is_frozen[name] = self.step_cnt < self.frozen_due[name]
            param.requires_grad = not self.is_frozen[name]
        return 
    
    def _threshold_update(self):
        return
