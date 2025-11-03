
from typing import Dict, List
import numpy as np
import torch
from torch.nn import Module
from torch.distributed.tensor import DTensor
from .action import ActionWithFreezing, ActionWithTime
from . import logger as pplog
from .schedule import adjust_freeze_ratio, gather_pipeline_schedule, set_freeze_ratio
from .config import TimelyFreezeConfig
from torchtitan.tools.logging import logger

class _Freezer:
    def __init__(self, model_parts: List[Module], config: TimelyFreezeConfig):
        '''Version 2: Updated on July 14, 2025
        - Collaborate with ActionWithFreezing to freeze per microbatch block.
        '''
        self.config : TimelyFreezeConfig = config
        self.stages : Dict[int, Module] = {stage: model_part for stage, model_part in zip(config.parallelism.stages_list, model_parts)}
        self.step_cnt = 0

        # Phases
        self.stability_check_freq = self.config.freezing.stability_check_freq
        '''Stability Check Frequency: check the stability every 10 steps (paper: 50)'''
        self.phase_unit = self.config.freezing.phase_unit
        '''Phase Unit. i.e., 1 epoch '''
        self.warmup_phase = max(self.phase_unit, self.config.lr_scheduler.warmup_steps - self.phase_unit)
        '''Warmup Phase: do nothing'''

        self.freeze_ratio_history = {stage_idx: [0] * (self.warmup_phase // self.stability_check_freq) if config.freezing.freeze else [] for stage_idx in self.stages.keys()} # frozen ratio history per stage
        self.paramwise_frozen_count = {stage_idx: {name: [0, 0] for name, _ in stage.named_parameters()} for stage_idx, stage in self.stages.items()} # [frozen, total] count for each layer in each stage
        return

    def _step_count(self, step:int):
        '''Count the number of steps and epochs. 
        Call this function at the start of each forward pass.
        Starting from (epoch 1, step 1) for the first microbatch.
        '''
        self.step_cnt = step
        return

    def freeze_update(self, step:int):
        self._step_count(step)
        if not self.config.freezing.freeze or self.step_cnt <= self.warmup_phase:
            return
        
        if self.step_cnt % self.stability_check_freq == 0:
            self.set_expected_freeze_ratio() # how many params to freeze ?
            self.set_params_to_freeze() # which params to freeze ?
            self.log_freeze_ratio() # log the current freeze ratio decision
        return

    def set_expected_freeze_ratio(self):
        raise NotImplementedError("This function should be implemented in the derived class.")

    def set_params_to_freeze(self):
        raise NotImplementedError("This function should be implemented in the derived class.")

    def log_freeze_ratio(self):
        '''Update the frozen ratio based on the current freezing status.'''
        if self.step_cnt % self.stability_check_freq != 0:
            return

        for stage_idx, stage in self.stages.items():
            for name, _ in stage.named_parameters():
                self.paramwise_frozen_count[stage_idx][name][0] += 0
                self.paramwise_frozen_count[stage_idx][name][1] += 1

            self.freeze_ratio_history[stage_idx].append(0)
        return

def get_freezer_class_version(freezer:_Freezer)->int:
    '''Get the version of the freezer class.'''
    if isinstance(freezer, _Freezer):
        return 1
    else:
        raise TypeError(f"Freezer should be an instance of _Freezer, but got {type(freezer)}.")

def get_freezer(model_parts: List[Module], config:TimelyFreezeConfig)->_Freezer:
    '''Get the freezer based on the metric type.'''
    if config.freezing.metric_type == 'fullrand7': # freeze per microbatch block - updated on Oct 18, 2025
        return FullyRandomFreezer_v7(model_parts, config)
    elif config.freezing.metric_type == 'apf': # APF (absolute perturbation freezing)
        return APFFreezer(model_parts, config)
    elif config.freezing.metric_type == 'timelyapf': # APF + Timely Freeze
        return APFFreezerWithTimelyFreeze(model_parts, config)
    elif config.freezing.metric_type == 'auto': # AutoFreeze
        return AutoFreezer(model_parts, config)
    elif config.freezing.metric_type == 'timelyauto': # AutoFreeze + Timely Freeze
        return AutoFreezerWithTimelyFreeze(model_parts, config)
    else:
        raise NotImplementedError(f"Metric Type [{config.freezing.metric_type}] is not supported.")



class FullyRandomFreezer_v7(_Freezer):
    def __init__(self, model_parts: List[Module], config: TimelyFreezeConfig):
        ''' Updated on Oct 18, 2025
        Set different expected freeze ratio per microbatch block.
        '''
        super().__init__(model_parts, config)

        self.progressive_freezing_phase = self.warmup_phase + 3 * self.phase_unit # ) if 'debug' in config.job.basename else (self.warmup_phase + config.lr_scheduler.warmup_epochs)
        '''Last step of progressive freezing phase: gradually increase the freezing_params_num to the expected number.'''
        self.progressive_freezing_start_step: int = -1
        '''Starting step of progressive freezing phase for pplog.'''
        self.monitoring_steps :int = min(self.phase_unit, 30)
        '''Number of steps for monitoring each of upperbound and lowerbound.'''
        self.freeze_adjust_freq = self.phase_unit
        '''Frequency of adjusting the freeze ratio during the stable freezing phase.'''
        
        # Phase Status
        self.monitoring_ub:bool = False 
        '''True if currently monitoring the upperbound of batch time, i.e., freezing ratio = 0.0 for all actions.'''
        self.monitoring_ub_start_step :int = -1
        '''Starting step of monitoring upperbound which is used in second monitoring phase for pplog.'''
        self.monitored_ub:bool = False
        '''True if monitored the upperbound of batch time. set to True at the first stability check freq after monitoring phase.'''
        self.monitoring_lb:bool = False
        '''True if currently monitoring the lowerbound of batch time, i.e., freezing ratio = 1.0 for all actions.'''
        self.monitoring_lb_start_step :int = -1
        '''Starting step of monitoring lowerbound which is used in second monitoring phase for pplog.'''
        self.monitored_lb:bool = False
        '''True if monitored the lowerbound of batch time, i.e., freezing ratio = 1.0 for all actions.'''

        self.pipeline_schedule :List[List[ActionWithFreezing]] = []
        '''Pipeline schedule with freezing information. Will be set after monitoring upper/lowerbound.'''
        # self.rand_noise_possibility = 0.05
        return
    
    def freeze_update(self, step:int):
        self._step_count(step)
        if not self.config.freezing.freeze or self.step_cnt <= self.warmup_phase:
            return

        if self.step_cnt % self.stability_check_freq == 0:
            self.set_expected_freeze_ratio()
            self.set_params_to_freeze() # calculate freezing metric -> set actual list of freezing params
            self.log_freeze_ratio()

            # log the current and expected freeze ratio per microbatch block
            if self.step_cnt % self.config.metrics.log_freq == 0 and \
                self.warmup_phase < self.step_cnt <= self.progressive_freezing_phase + 2 * self.phase_unit and \
                    self.monitored_ub and self.monitored_lb:
                logger.info(f"Current/Expected Freeze Ratio per Block: {', '.join([f'[MB{action.microbatch}] {action.actual_freeze_ratio:.2f}/{action.expected_freeze_ratio:.2f}' for action in self.pipeline_schedule[self.config.comm.pp_rank] if action.freezable])}")
        return
    
    def set_expected_freeze_ratio(self):
        '''Set the expected freeze ratio based on the backward time.'''
        # Warmup + Monitoring Phase
        if self.step_cnt <= self.warmup_phase: # Warmup Phase
            # during the monitoring phase, do not freeze the model
            self.monitored_ub = False

        elif not (self.monitoring_lb or self.monitored_lb or self.monitoring_ub or  self.monitored_ub):
            self._start_monitoring_upperbound()
        
        elif self.monitoring_ub: # monitoring upperbound
            if pplog.pipeline_log.step_cnt > self.monitoring_ub_start_step + self.monitoring_steps :
                self._set_upperbound()
                self._start_monitoring_lowerbound()
            else:
                logger.warning(f"[Step {self.step_cnt}] ⏳ Monitoring Upperbound... ({pplog.pipeline_log.step_cnt}/{self.monitoring_ub_start_step + self.monitoring_steps})")

        elif self.monitoring_lb: # monitoring lowerbound
            if len(pplog.pipeline_log.log_schedule[0].log_duration) > self.monitoring_lb_start_step + self.monitoring_steps :
                self._set_lowerbound()
        
        # start progressive freezing phase
        elif self.step_cnt <= self.progressive_freezing_phase:
            # during the warmup phase, gradually increase the progressive_freezing
            for a, la in zip(self.pipeline_schedule[self.config.comm.pp_rank], pplog.pipeline_log.log_schedule):
                a.progressive_freezing = (self.step_cnt-self.warmup_phase)/(self.progressive_freezing_phase-self.warmup_phase)
            
        # Stable Freezing Phase - periodically adjust the freeze ratio 
        elif self.step_cnt > self.progressive_freezing_phase + self.freeze_adjust_freq and self.step_cnt % self.freeze_adjust_freq == 0: 
            monitored_values_dict = {stage: [] for stage in self.stages.keys()}
            for a, la in zip(self.pipeline_schedule[self.config.comm.pp_rank], pplog.pipeline_log.log_schedule):
                if not a.freezable:
                    continue
                times = la.log_duration[self.progressive_freezing_start_step:]
                time_outliers = (np.percentile(times, 5), np.percentile(times, 95))
                afrs = a.freeze_ratio_history[self.progressive_freezing_start_step:self.progressive_freezing_start_step + len(times)]
                monitored_values_dict[a.stage] += [(afr, time) for (afr, time) in zip(afrs, times) if time_outliers[0]<=time<=time_outliers[1]] 
            self.pipeline_schedule = adjust_freeze_ratio(self.pipeline_schedule, monitored_values_dict, self.config)
            logger.info(f"Adjusted Freeze Ratio per Block: {', '.join([f'[MB{action.microbatch}] {action.actual_freeze_ratio:.2f}/{action.expected_freeze_ratio:.2f}' for action in self.pipeline_schedule[self.config.comm.pp_rank] if action.freezable])}")
            self.freeze_adjust_freq = self.freeze_adjust_freq * 2 # stop adjusting the freeze ratio        
        else:
            pass
        return
    
    
    def set_params_to_freeze(self):
        '''Random Selection of parameters to freeze based on the expected freeze ratio.'''
        if len(self.pipeline_schedule) == 0:
            return
        
        for action in self.pipeline_schedule[self.config.comm.pp_rank]:
            actual_num_freeze = min(action.num_params, int(np.round(action.num_params * action.actual_freeze_ratio)))
            if actual_num_freeze <= 0:
                action.freezing_list = [False] * action.num_params
            else:
                if action.stage == 0: # front layers more likely to freeze
                    weights = torch.linspace(1.0, 0.1, steps=action.num_params)
                    idx = torch.multinomial(weights, actual_num_freeze, replacement=False)
                else:
                    idx = torch.randperm(action.num_params)[:actual_num_freeze]
                freezing_list = torch.zeros(action.num_params, dtype=torch.bool)
                freezing_list[idx] = True
                action.freezing_list = freezing_list.tolist()
        return 
    

    def log_freeze_ratio(self):
        '''Update the frozen ratio based on the current freezing status.'''
        if self.step_cnt % self.stability_check_freq != 0:
            return

        for stage_idx, stage in self.stages.items():
            if self.monitored_ub:
                for name, _ in stage.named_parameters():
                    self.paramwise_frozen_count[stage_idx][name][0] = sum([a.paramwise_frozen_count[name][0] for a in self.pipeline_schedule[self.config.comm.pp_rank] if a.stage == stage_idx and name in a.paramwise_frozen_count])
                    self.paramwise_frozen_count[stage_idx][name][1] = sum([a.paramwise_frozen_count[name][1] for a in self.pipeline_schedule[self.config.comm.pp_rank] if a.stage == stage_idx and name in a.paramwise_frozen_count])

                average_freeze_ratio = float(np.mean([a.actual_freeze_ratio for a in self.pipeline_schedule[self.config.comm.pp_rank] if a.stage == stage_idx and a.freezable]))
                self.freeze_ratio_history[stage_idx].append(average_freeze_ratio)
            else:
                self.freeze_ratio_history[stage_idx].append(0)

        return
    
    def _start_monitoring_upperbound(self):
        '''Set freeze ratio = 0.0 for all actions and start monitoring the upperbound of batch time.'''
        assert not (self.monitored_ub or self.monitoring_ub), "Upperbound monitoring has already been started or finished."
        if self.config.comm.is_master_rank:
            logger.info(f"[Step {self.step_cnt}] 〰️ Monitoring Upperbound")
        self.monitored_ub = False
        self.monitoring_ub = True
        self.monitoring_ub_start_step = pplog.pipeline_log.step_cnt
        return

    def _set_upperbound(self):
        '''Create a pipeline schedule with freeze ratio = 0.0 for all actions.'''
        assert self.monitoring_ub and not self.monitored_ub, "Upperbound monitoring has not been started or has already been finished."
        if self.config.comm.is_master_rank:
            logger.info(f"[Step {self.step_cnt}] ✔️  Setting Upperbound")
        # create upperbound pipeline schedule
        pipeline_schedule_tmp :List[List[ActionWithTime]] = gather_pipeline_schedule(pplog.pipeline_log.log_schedule, comm=self.config.comm, log_window=self.monitoring_steps)
        self.pipeline_schedule = [[ActionWithFreezing(action.type, action.rank, action.microbatch, action.stage, action.duration) \
                                                                for action in rank_actions] for rank_actions in pipeline_schedule_tmp]
        # Set the stage module for each action in the pipeline schedule
        stage_dict = {stage_idx: stage for stage_idx, stage in self.stages.items()}
        for a in self.pipeline_schedule[self.config.comm.pp_rank]:
            a.module = stage_dict[a.stage]
            a.freeze_flag = True
        pplog.pipeline_log.action_dict = {(action.type, action.rank, action.microbatch, action.stage): action \
                                            for action in self.pipeline_schedule[self.config.comm.pp_rank]}

        self.monitored_ub = True
        self.monitoring_ub = False
        return

    def _start_monitoring_lowerbound(self):
        '''Freeze all parameters and start monitoring the lowerbound of batch time.'''
        assert not (self.monitored_lb or self.monitoring_lb), "Lowerbound monitoring has already been started or finished."
        if self.config.comm.is_master_rank:
            logger.info(f"[Step {self.step_cnt}] 〰️ Monitoring Lowerbound")
        for a in self.pipeline_schedule[self.config.comm.pp_rank]:
            a.progressive_freezing = 1.0
            if a.stage == self.config.parallelism.num_stages - 1: # last stage
                a.expected_freeze_ratio = 1.0 - 1/a.num_params
            else:
                a.expected_freeze_ratio = 1.0
        
        self.monitored_lb = False
        self.monitoring_lb = True
        self.monitoring_lb_start_step = pplog.pipeline_log.step_cnt
        return

    def _set_lowerbound(self):
        '''Set the min_duration of each action block based on the monitored lowerbound of batch time.'''
        assert self.monitoring_lb and not self.monitored_lb, "Lowerbound monitoring has not been started or has already been finished."
        assert self.monitoring_lb_start_step >= 0, "Lowerbound monitoring start step is not set."
        if self.config.comm.is_master_rank:
            logger.info(f"[Step {self.step_cnt}] ✔️  Setting Lowerbound")
        # create lowerbound pipeline schedule
        log_window = len(pplog.pipeline_log.log_schedule[-1].log_duration) - self.monitoring_lb_start_step
        pipeline_schedule_lb :List[List[ActionWithTime]] = gather_pipeline_schedule(pplog.pipeline_log.log_schedule, comm=self.config.comm, log_window=log_window)

        # set the min_duration of each action block based on the lowerbound log time
        for ar_lb, actions_per_rank in zip(pipeline_schedule_lb, self.pipeline_schedule):
            for a_lb, a in zip(ar_lb, actions_per_rank):
                if a.freezable:
                    a.min_duration = a_lb.duration

        self.pipeline_schedule = set_freeze_ratio(self.pipeline_schedule, config=self.config)

        self.monitored_lb = True
        self.monitoring_lb = False
        self.progressive_freezing_start_step = pplog.pipeline_log.step_cnt
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
    def __init__(self, model_parts: List[Module], config: TimelyFreezeConfig):
        super().__init__(model_parts, config)

        # freezing metrics
        self.alpha = 0.99 # parameter for exponential moving average (paper: 0.99)
        self.last_param = {name: None for stage in self.stages.values() for name, _ in stage.named_parameters()} # cumulative update
        self.ema = {name: 0.0 for stage in self.stages.values() for name, _ in stage.named_parameters()}
        self.ema_abs = {name: 0.0 for stage in self.stages.values() for name, _ in stage.named_parameters()}

        # freeze update
        self.threshold = getattr(config.freezing, "threshold", 0.05) # threshold on effective perturbation. (paper: 0.05)
        self.freezing_period = {name: 0 for stage in self.stages.values() for name, _ in stage.named_parameters()} # freeze the layer for a unit of stability_check_freq (paper: 50)
        self.frozen_due = {name: 0 for stage in self.stages.values() for name, _ in stage.named_parameters()}
        self.is_frozen = {name: False for stage in self.stages.values() for name, _ in stage.named_parameters()}
        self.freeze_ratio = {stage_idx: 0 for stage_idx in self.stages.keys()}
        return

    def freeze_update(self, step:int):
        self._step_count(step)
        if not self.config.freezing.freeze or self.step_cnt <= self.warmup_phase:
            return

        if self.step_cnt % self.stability_check_freq == 0:
            self.set_params_to_freeze()
            self.log_freeze_ratio()
            logger.info(f"Freezing Ratio: {', '.join([f'[Stage{stage_idx}] {self.freeze_ratio[stage_idx]:.2f}' for stage_idx in self.stages])}, Threshold: {self.threshold:.4e}")
        return
    
    def set_params_to_freeze(self):
        '''Update the freezing status of the model parameters based on the threshold.'''
        freezing_metric :Dict[str, float] = self._calculate_params_metric()

        for stage in self.stages.values():
            for name, param in stage.named_parameters():
                if not self.is_frozen[name]:
                    # update the freezing period (update in TCP style)
                    if freezing_metric[name] <= self.threshold: # stable -> increase the freezing period
                        self.freezing_period[name] += self.stability_check_freq
                    else: # unstable -> decrease the freezing period
                        self.freezing_period[name] = self.freezing_period[name]/2
                        if self.freezing_period[name] < self.stability_check_freq:
                            self.freezing_period[name] = 0

                    # update the freezing deadline
                    self.frozen_due[name] = self.step_cnt + self.freezing_period[name]

                # update the freezing status
                self.is_frozen[name] = self.step_cnt < self.frozen_due[name]
                param.requires_grad = not self.is_frozen[name]
        return 
    
    def _calculate_params_metric(self) -> Dict[str, float]:
        '''Calculate and return the effective perturbation metric for parameters.'''
        freezing_metric = {name: 99999999999 for stage in self.stages.values() for name, _ in stage.named_parameters()} # effective perturbation metric. -> 0: oscillation (stable), -> 1: directional (unstable)

        for stage in self.stages.values():
            for name, param in stage.named_parameters():
                if param.grad is None:
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
                    freezing_metric[name] = abs(self.ema[name]) / max(self.ema_abs[name], 1e-10)
                self.last_param[name] = curr_param  # reset the cumulative update after calculating the metric
        return freezing_metric

    
    def log_freeze_ratio(self):
        '''Update the frozen ratio based on the current freezing status.'''
        if self.step_cnt % self.stability_check_freq != 0:
            return

        for stage_idx, stage in self.stages.items():
            cnt_0, cnt_1 = 0, 0
            for name, _ in stage.named_parameters():
                cnt_0, cnt_1 = cnt_0 + self.is_frozen[name], cnt_1 + 1
                self.paramwise_frozen_count[stage_idx][name][0] += self.is_frozen[name]
                self.paramwise_frozen_count[stage_idx][name][1] += 1

            self.freeze_ratio[stage_idx] = cnt_0 / cnt_1
            self.freeze_ratio_history[stage_idx].append(self.freeze_ratio[stage_idx])
        return
    

class AutoFreezer(_Freezer):
    '''
    ** Baseline Paper : [Preprint] AutoFreeze: Automatically Freezing Model Blocks to Accelerate Fine-tuning
        - https://arxiv.org/abs/2102.01386

    Update the freezing status based on the absolute perturbation freezing (APF).
    - Metric =  |grad_t - grad_{t-1}| / grad_{t-1}
    - Threshold = percentile(grad_changes, p)
    - Layers whose metric >= threshold are frozen.
    '''
    def __init__(self, model_parts: List[Module], config: TimelyFreezeConfig):
        super().__init__(model_parts, config)

        # freezing metrics
        self.prev_grad_norm = {name: None for stage in self.stages.values() for name, _ in stage.named_parameters()}
        self.percentile = getattr(config.freezing, "percentile", 50)
        self.threshold = 0.0 # will be set automatically based on the percentile
        self.start_layer = 0

        # freeze update
        self.is_frozen = {name: False for stage in self.stages.values() for name, _ in stage.named_parameters()}
        self.freeze_ratio = {stage_idx: 0 for stage_idx in self.stages.keys()}
        return

    def freeze_update(self, step:int):
        self._step_count(step)
        if not self.config.freezing.freeze or self.step_cnt <= self.warmup_phase:
            return

        if self.step_cnt % self.stability_check_freq == 0:
            self.set_params_to_freeze()
            self.log_freeze_ratio()
            logger.info(f"Freezing Ratio: {', '.join([f'[Stage{stage_idx}] {self.freeze_ratio[stage_idx]:.2f}' for stage_idx, stage in self.stages.items()])}, Threshold: {self.threshold:.4e}")
        return
    
    
    def _get_layer_index(self, name: str):
        """Extract layer number from parameter name, e.g., 'encoder.layer.5.attention...' → 5."""
        for part in name.replace('_', '.').split('.'):
            if part.isdigit():
                return int(part)
        return None
    
    def _calculate_params_metric(self) -> Dict[int, float]:
        """Accumulate gradient norms for each parameter."""
        curr_grad_norm = {name: 0.0 for stage in self.stages.values() for name, _ in stage.named_parameters()}
        for stage in self.stages.values():
            for name, param in stage.named_parameters():
                if param.grad is not None:
                    curr_grad_norm[name] += torch.norm(param.grad.detach(), p=1).item()

        layer_grad_change = {}
        for stage in self.stages.values():
            for name, _ in stage.named_parameters():
                layer_idx = self._get_layer_index(name)
                if layer_idx is None:
                    continue
                prev = self.prev_grad_norm.get(name, None)
                curr = curr_grad_norm[name]
                layer_grad_change.setdefault(layer_idx, 0)
                if prev is not None and prev > 0:
                    change_ratio = abs(prev - curr) / prev
                    layer_grad_change[layer_idx] += change_ratio

        # reset grad accumulation
        self.prev_grad_norm = curr_grad_norm
        return layer_grad_change
    
    def set_params_to_freeze(self):
        """Decide which layers to freeze based on gradient variation."""
        metric_per_layer = self._calculate_params_metric()
        # percentile-based threshold
        if not metric_per_layer:
            return
        self.threshold = np.percentile([v for k, v in metric_per_layer.items() if k >= self.start_layer], self.percentile)

        # choose the first layer that exceeds the threshold
        for layer_idx, ratio in metric_per_layer.items():
            if ratio >= self.threshold:
                self.start_layer = layer_idx
                break

        # update freezing mask
        for stage in self.stages.values():
            for name, param in stage.named_parameters():
                layer_idx = self._get_layer_index(name)
                self.is_frozen[name] = layer_idx is not None and layer_idx < self.start_layer
                param.requires_grad = not self.is_frozen[name]
        return
    
    def log_freeze_ratio(self):
        '''Update the frozen ratio based on the current freezing status.'''
        if self.step_cnt % self.stability_check_freq != 0:
            return

        for stage_idx, stage in self.stages.items():
            cnt_0, cnt_1 = 0, 0
            for name, _ in stage.named_parameters():
                cnt_0, cnt_1 = cnt_0 + self.is_frozen[name], cnt_1 + 1
                self.paramwise_frozen_count[stage_idx][name][0] += self.is_frozen[name]
                self.paramwise_frozen_count[stage_idx][name][1] += 1

            self.freeze_ratio[stage_idx] = cnt_0 / cnt_1
            self.freeze_ratio_history[stage_idx].append(self.freeze_ratio[stage_idx])
        return
    


class APFFreezerWithTimelyFreeze(FullyRandomFreezer_v7):
    def __init__(self, model_parts: List[Module], config: TimelyFreezeConfig):
        ''' Updated on Oct 18, 2025
        TimelyFreeze + AutoFreeze (follow the freeze ratio of timelyfreeze, but primarily use AutoFreeze metric to select which params to freeze)
        '''
        super().__init__(model_parts, config)

        # freezing metrics
        self.alpha = 0.99 # parameter for exponential moving average (paper: 0.99)
        self.last_param = {name: None for stage in self.stages.values() for name, _ in stage.named_parameters()} # cumulative update
        self.ema = {name: 0.0 for stage in self.stages.values() for name, _ in stage.named_parameters()}
        self.ema_abs = {name: 0.0 for stage in self.stages.values() for name, _ in stage.named_parameters()}

        # freeze update
        self.threshold = getattr(config.freezing, "threshold", 0.05) # threshold on effective perturbation. (paper: 0.05)
        self.freezing_period = {name: 0 for stage in self.stages.values() for name, _ in stage.named_parameters()} # freeze the layer for a unit of stability_check_freq (paper: 50)
        self.frozen_due = {name: 0 for stage in self.stages.values() for name, _ in stage.named_parameters()}
        return
    
    
    
    def freeze_update(self, step:int):
        self._step_count(step)
        if not self.config.freezing.freeze or self.step_cnt <= self.warmup_phase:
            return

        if self.step_cnt % self.stability_check_freq == 0:
            self.set_expected_freeze_ratio()
            self.set_params_to_freeze() # calculate freezing metric -> set actual list of freezing params
            self.log_freeze_ratio()

            # log the current and expected freeze ratio per microbatch block
            if self.step_cnt % self.config.metrics.log_freq == 0 and \
                self.warmup_phase < self.step_cnt <= self.progressive_freezing_phase + 2 * self.phase_unit and \
                    self.monitored_ub and self.monitored_lb:
                logger.info(f"Current/Expected Freeze Ratio per Block: {', '.join([f'[MB{action.microbatch}] {action.actual_freeze_ratio:.2f}/{action.expected_freeze_ratio:.2f}' for action in self.pipeline_schedule[self.config.comm.pp_rank] if action.freezable])}")
        return

    def set_expected_freeze_ratio(self):
        '''Set the expected freeze ratio based on the backward time.'''
        super().set_expected_freeze_ratio()
        return

    def set_params_to_freeze(self):
        '''Update the freezing status of the model parameters based on the threshold.'''
        freezing_metric :Dict[str, float] = self._calculate_params_metric()
        freeze_cand :Dict[int, List[bool]] = {} # whether the parameter's metric is candidate for freezing
        for stage_idx, stage in self.stages.items():
            freeze_cand[stage_idx] = []
            for name, param in stage.named_parameters():
                # update the freezing period (update in TCP style)
                if freezing_metric[name] <= self.threshold:
                    self.freezing_period[name] += self.stability_check_freq
                else:
                    self.freezing_period[name] = self.freezing_period[name]/2
                if self.freezing_period[name] < self.stability_check_freq:
                    self.freezing_period[name] = 0

                # update the freezing deadline
                self.frozen_due[name] = self.step_cnt + self.freezing_period[name]

                # update the freezing candidates
                freeze_cand[stage_idx].append(self.step_cnt < self.frozen_due[name])

        if len(self.pipeline_schedule) == 0:
            return

        for action in self.pipeline_schedule[self.config.comm.pp_rank]:
            actual_num_freeze = min(action.num_params, int(np.round(action.num_params * action.actual_freeze_ratio)))
            if actual_num_freeze > 0:
                weights = [1 if val else 0.01 for val in freeze_cand[action.stage]]
                assert len(weights) == action.num_params, f"Length Mismatch: {len(weights)} vs {action.num_params}"
                idx = torch.multinomial(torch.tensor(weights, dtype=torch.float16), actual_num_freeze, replacement=False)
                freezing_list = torch.zeros(action.num_params, dtype=torch.bool)
                freezing_list[idx] = True
                action.freezing_list = freezing_list.tolist()
            else:
                action.freezing_list = [False] * action.num_params
        return 

    def log_freeze_ratio(self):
        '''Update the frozen ratio based on the current freezing status.'''
        if self.step_cnt % self.stability_check_freq != 0:
            return

        for stage_idx, stage in self.stages.items():
            if self.monitored_ub:
                for name, _ in stage.named_parameters():
                    self.paramwise_frozen_count[stage_idx][name][0] = sum([a.paramwise_frozen_count[name][0] for a in self.pipeline_schedule[self.config.comm.pp_rank] if a.stage == stage_idx and name in a.paramwise_frozen_count])
                    self.paramwise_frozen_count[stage_idx][name][1] = sum([a.paramwise_frozen_count[name][1] for a in self.pipeline_schedule[self.config.comm.pp_rank] if a.stage == stage_idx and name in a.paramwise_frozen_count])

                average_freeze_ratio = float(np.mean([a.actual_freeze_ratio for a in self.pipeline_schedule[self.config.comm.pp_rank] if a.stage == stage_idx and a.freezable]))
                self.freeze_ratio_history[stage_idx].append(average_freeze_ratio)
            else:
                self.freeze_ratio_history[stage_idx].append(0)
        return
    

    def _calculate_params_metric(self) -> Dict[str, float]:
        '''Calculate and return the effective perturbation metric for parameters.'''
        freezing_metric = {name: 99999999999 for stage in self.stages.values() for name, _ in stage.named_parameters()} # effective perturbation metric. -> 0: oscillation (stable), -> 1: directional (unstable)

        for stage in self.stages.values():
            for name, param in stage.named_parameters():
                if param.grad is None:
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
                    freezing_metric[name] = abs(self.ema[name]) / max(self.ema_abs[name], 1e-10)
                self.last_param[name] = curr_param  # reset the cumulative update after calculating the metric
        return freezing_metric


class AutoFreezerWithTimelyFreeze(FullyRandomFreezer_v7):
    def __init__(self, model_parts: List[Module], config: TimelyFreezeConfig):
        ''' Updated on Oct 26, 2025
        TimelyFreeze + APF (follow the freeze ratio of timelyfreeze, but primarily use APF metric to select which params to freeze)
        '''
        super().__init__(model_parts, config)

        # freezing metrics
        self.prev_grad_norm = {name: None for stage in self.stages.values() for name, _ in stage.named_parameters()}
        self.percentile = getattr(config.freezing, "percentile", 50)
        self.threshold = 0.0
        self.start_layer = 0

        # freeze update
        self.is_frozen = {name: False for stage in self.stages.values() for name, _ in stage.named_parameters()}
        self.freeze_ratio = {stage_idx: 0 for stage_idx in self.stages.keys()}
        return

    def freeze_update(self, step:int):
        self._step_count(step)
        if not self.config.freezing.freeze or self.step_cnt <= self.warmup_phase:
            return

        if self.step_cnt % self.stability_check_freq == 0:
            self.set_expected_freeze_ratio()
            self.set_params_to_freeze() # calculate freezing metric -> set actual list of freezing params
            self.log_freeze_ratio()

            # log the current and expected freeze ratio per microbatch block
            if self.step_cnt % self.config.metrics.log_freq == 0 and \
                self.warmup_phase < self.step_cnt <= self.progressive_freezing_phase + 2 * self.phase_unit and \
                    self.monitored_ub and self.monitored_lb:
                logger.info(f"Current/Expected Freeze Ratio per Block: {', '.join([f'[MB{action.microbatch}] {action.actual_freeze_ratio:.2f}/{action.expected_freeze_ratio:.2f}' for action in self.pipeline_schedule[self.config.comm.pp_rank] if action.freezable])}")
        return
    
    
    def set_expected_freeze_ratio(self):
        '''Set the expected freeze ratio based on the backward time.'''
        super().set_expected_freeze_ratio()
        return
    
    def set_params_to_freeze(self):
        """Decide which layers to freeze based on gradient variation."""
        metric_per_layer :Dict[int, float] = self._calculate_params_metric()
        freeze_cand :Dict[int, List[bool]] = {} # whether the parameter's metric is candidate for freezing
        # percentile-based threshold
        if not metric_per_layer:
            return
        self.threshold = np.percentile([v for k, v in metric_per_layer.items() if k >= self.start_layer], self.percentile)

        # choose the first layer that exceeds the threshold
        for layer_idx, ratio in metric_per_layer.items():
            if ratio >= self.threshold:
                self.start_layer = layer_idx
                break

        # update the freezing candidates
        for stage_idx, stage in self.stages.items():
            freeze_cand[stage_idx] = []
            for name, param in stage.named_parameters():
                layer_idx = self._get_layer_index(name)
                freeze_cand[stage_idx].append(layer_idx is not None and layer_idx < self.start_layer)

        if len(self.pipeline_schedule) == 0:
            return
        
        for action in self.pipeline_schedule[self.config.comm.pp_rank]:
            actual_num_freeze = min(action.num_params, int(np.round(action.num_params * action.actual_freeze_ratio)))
            if actual_num_freeze > 0:
                weights = [1 if val else 0.01 for val in freeze_cand[action.stage]]
                assert len(weights) == action.num_params, f"Length Mismatch: {len(weights)} vs {action.num_params}"
                idx = torch.multinomial(torch.tensor(weights, dtype=torch.float16), actual_num_freeze, replacement=False)
                freezing_list = torch.zeros(action.num_params, dtype=torch.bool)
                freezing_list[idx] = True
                action.freezing_list = freezing_list.tolist()
            else:
                action.freezing_list = [False] * action.num_params
        return 
    
    def log_freeze_ratio(self):
        '''Update the frozen ratio based on the current freezing status.'''
        if self.step_cnt % self.stability_check_freq != 0:
            return

        for stage_idx, stage in self.stages.items():
            if self.monitored_ub:
                for name, _ in stage.named_parameters():
                    self.paramwise_frozen_count[stage_idx][name][0] = sum([a.paramwise_frozen_count[name][0] for a in self.pipeline_schedule[self.config.comm.pp_rank] if a.stage == stage_idx and name in a.paramwise_frozen_count])
                    self.paramwise_frozen_count[stage_idx][name][1] = sum([a.paramwise_frozen_count[name][1] for a in self.pipeline_schedule[self.config.comm.pp_rank] if a.stage == stage_idx and name in a.paramwise_frozen_count])

                average_freeze_ratio = float(np.mean([a.actual_freeze_ratio for a in self.pipeline_schedule[self.config.comm.pp_rank] if a.stage == stage_idx and a.freezable]))
                self.freeze_ratio_history[stage_idx].append(average_freeze_ratio)
            else:
                self.freeze_ratio_history[stage_idx].append(0)
        return
    
    def _get_layer_index(self, name: str):
        """Extract layer number from parameter name, e.g., 'encoder.layer.5.attention...' → 5."""
        for part in name.replace('_', '.').split('.'):
            if part.isdigit():
                return int(part)
        return None
    
    def _calculate_params_metric(self) -> Dict[int, float]:
        """Accumulate gradient norms for each parameter."""
        curr_grad_norm = {name: 0.0 for stage in self.stages.values() for name, _ in stage.named_parameters()}
        for stage in self.stages.values():
            for name, param in stage.named_parameters():
                if param.grad is not None:
                    curr_grad_norm[name] += torch.norm(param.grad.detach(), p=1).item()

        layer_grad_change = {}
        for stage in self.stages.values():
            for name, _ in stage.named_parameters():
                layer_idx = self._get_layer_index(name)
                if layer_idx is None:
                    continue
                prev = self.prev_grad_norm.get(name, None)
                curr = curr_grad_norm[name]
                layer_grad_change.setdefault(layer_idx, 0)
                if prev is not None and prev > 0:
                    change_ratio = abs(prev - curr) / prev
                    layer_grad_change[layer_idx] += change_ratio

        # reset grad accumulation
        self.prev_grad_norm = curr_grad_norm
        return layer_grad_change
    

