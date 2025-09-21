
from typing import Dict, List
import numpy as np
import torch
from torch.nn import Module
from .action import ActionWithFreezing, ActionWithTime
from .logger import pipeline_logger
from .schedule import adjust_freeze_ratio, gather_pipeline_schedule, set_freeze_ratio
from .config import TimelyFreezeConfig
from torchtitan.tools.logging import logger

class _Freezer:
    def __init__(self, model_parts: List[Module], config: TimelyFreezeConfig):
        '''Version 2: Updated on July 14, 2025
        - Collaborate with ActionWithFreezing to freeze per microbatch block.
        '''

        self.config : TimelyFreezeConfig = config
        self.pp_rank : int = config.comm.pp_rank # current rank
        assert len(model_parts) == len(config.parallelism.stages_list), f"Number of model parts {len(model_parts)} and stages {len(config.parallelism.stages_list)} should be the same."
        self.stages : Dict[int, Module] = {stage: model_part for stage, model_part in zip(config.parallelism.stages_list, model_parts)}
    
        # Phases
        self.stability_check_freq = self.config.freezing.stability_check_freq
        '''Stability Check Frequency: check the stability every 10 steps'''
        self.phase_unit = self.config.freezing.phase_unit
        '''Phase Unit. i.e., 100 steps'''
        self.warmup_phase = self.phase_unit
        '''Warmup Phase: do nothing'''
        self.monitoring_phase = self.warmup_phase + self.phase_unit
        '''Monitoring Phase: do not freeze the model, only analyze the time. At least 1 phase unit for monitoring'''
        assert self.phase_unit % self.stability_check_freq == 0, "Phase unit should be a multiple of stability check frequency."

        # Logging
        self.logger = pipeline_logger.initialize(config) # logger for the pipeline schedule
        self.freeze_ratio_history = {stage_idx: [0] * (self.monitoring_phase // self.stability_check_freq) if config.freezing.freeze else [] for stage_idx in self.stages.keys()} # frozen ratio history per stage
        self.paramwise_frozen_count = {stage_idx: {name: [0, 0] for name, _ in stage.named_parameters()} for stage_idx, stage in self.stages.items()} # [frozen, total] count for each layer in each stage
        return

    def freeze_update(self):
        curr_step :int = self.logger.step_cnt
        if curr_step % self.stability_check_freq == 0:
            self._set_expected_freeze_ratio()
            self._log_freeze_ratio()

            last_efr_per_stage = {s: f'{self.freeze_ratio_history[s][-1]:.2f}' for s in self.freeze_ratio_history.keys()}
            logger.info(f"Freezing Ratio: {last_efr_per_stage}, Threshold: {self.threshold:.4e}")
        return
    
    def _set_expected_freeze_ratio(self):
        pass
        # raise NotImplementedError("This function should be implemented in the derived class.")

    def _update_freeze_ratio(self):
        '''Update the frozen ratio based on the current freezing status.'''
        if self.logger.step_cnt % self.stability_check_freq != 0:
            return

        for stage_idx, stage in self.stages.items():
            for name, _ in stage.named_parameters():
                self.paramwise_frozen_count[stage_idx][name][0] += self.is_frozen[name]
                self.paramwise_frozen_count[stage_idx][name][1] += 1

            self.freeze_ratio_history[stage_idx].append(sum([v[0] for v in self.paramwise_frozen_count[stage_idx].values()]) / len(self.paramwise_frozen_count[stage_idx]))
        return


def get_freezer(model: List[Module], config:TimelyFreezeConfig)->_Freezer:
    '''Get the freezer based on the metric type.'''
    if config.freezing.metric_type == 'fullrand6': # freeze per microbatch block - updated on July 14, 2025
        return FullyRandomFreezer_v6(model, config)
    elif config.freezing.metric_type == 'apf': # APF (absolute perturbation freezing)
        return APFFreezer(model, config)
    elif config.freezing.metric_type == 'timelyapf': # APF + Timely Freeze
        return APFFreezerWithTimelyFreeze(model, config)
    else:
        raise NotImplementedError(f"Metric Type [{config.freezing.metric_type}] is not supported.")


class FullyRandomFreezer_v6(_Freezer):
    def __init__(self, model_parts: List[Module], config: TimelyFreezeConfig):
        ''' Updated on July 14, 2025
        Set different expected freeze ratio per microbatch block.
        '''
        super().__init__(model_parts, config)

        self.progressive_freezing_phase = self.monitoring_phase + self.config.lr_scheduler.warmup_steps 
        '''Progressive Freezing Phase: gradually increase the freezing_params_num to the expected number.'''

        self.monitored_ub = False
        '''True if monitored the upperbound of batch time. set to True at the first stability check freq after monitoring phase.'''
        self.monitoring_lb = False 
        '''True if currently monitoring the lowerbound of batch time, i.e., freezing ratio = 1.0 for all actions.'''
        self.monitored_lb = False
        '''True if monitored the lowerbound of batch time, i.e., freezing ratio = 1.0 for all actions.'''
        self.monitoring_lb_start = None
        '''Starting step of monitoring lowerbound which is used in second monitoring phase.'''

        self.freeze_adjust_freq = self.phase_unit
        self.rand_noise_possibility = 0.1
        self.progressive_freezing_start_step = None
        self.pipeline_schedule:List[List[ActionWithFreezing]] = [] # the pipeline schedule, which is a list of list of ActionWithFreezing
        return

    def freeze_update(self):
        curr_step :int = self.logger.step_cnt
        if not self.config.freezing.freeze or curr_step <= self.warmup_phase:
            return

        if curr_step % self.stability_check_freq == 0:
            self._set_expected_freeze_ratio()

            if self.monitoring_phase < curr_step:
                self._log_freeze_ratio()

        # log the current and expected freeze ratio per microbatch block
        if curr_step % self.config.metrics.log_freq == 0 and self.monitoring_phase < curr_step < self.progressive_freezing_phase + 2 * self.phase_unit:
            logger.info(f"Current/Expected Freeze Ratio per Block: {', '.join([f'[MB{a.microbatch}] {a.actual_freeze_ratio:.2f}/{a.expected_freeze_ratio:.2f}' for a in self.pipeline_schedule[self.pp_rank] if a.freezable])}")
        return

    def _set_expected_freeze_ratio(self):
        '''Set the expected freeze ratio based on the backward time.'''
        curr_step :int = self.logger.step_cnt
        if curr_step <= self.monitoring_phase: 
            # during the monitoring phase, do not freeze the model
            self.monitored_ub = False
            return
        elif curr_step < self.progressive_freezing_phase:
            if not self.monitored_ub: # first stability check freq after monitoring phase
                pipeline_schedule :List[List[ActionWithFreezing]] = set_freeze_ratio(gather_pipeline_schedule(self.logger.rank_schedule.schedule, self.config.comm), self.config)

                # Set the stage module for each action in the pipeline schedule
                for a in pipeline_schedule[self.pp_rank]:
                    a.module = self.stages[a.stage]
                    a.freeze_flag = True
                self.pipeline_schedule = pipeline_schedule
                self.logger.action_dict = {(action.type, action.rank, action.microbatch, action.stage): action \
                                                    for action in pipeline_schedule[self.pp_rank]}
                self.monitored_ub = True
                self.progressive_freezing_start_step = curr_step

            # during the warmup phase, gradually increase the progressive_freezing
            for a, la in zip(self.pipeline_schedule[self.pp_rank], self.logger.rank_schedule):
                a.progressive_freezing = (curr_step-self.monitoring_phase)/(self.progressive_freezing_phase-self.monitoring_phase)
            return
        # End of the Progressive Freezing Phase
        # during the last 10 steps of the progressive freezing phase, monitor the lowerbound of batch time and set min_duration of each action block
        elif curr_step == self.progressive_freezing_phase: 
            # 1) Monitor the lowerbound of batch time
            if not self.monitoring_lb and not self.monitored_lb: # start lowerbound monitoring phase
                for a in self.pipeline_schedule[self.pp_rank]:
                    a.progressive_freezing = 1.0
                    if a.stage == self.config.parallelism.num_stages - 1: # last stage
                        a.expected_freeze_ratio = 1.0 - 1/a.num_params
                    else:
                        a.expected_freeze_ratio = 1.0
                self.monitoring_lb = True
                self.monitoring_lb_start = curr_step
                self.rand_noise_possibility = 0
            # 2) Set min_duration of each action block
            elif not self.monitored_lb and self.monitoring_lb \
                and len(self.logger.rank_schedule[0].log_time) > self.monitoring_lb_start + 20 :
                # create lowerbound pipeline schedule
                log_window = len(self.logger.rank_schedule[-1].log_time) - self.monitoring_lb_start
                pipeline_schedule_lb :List[List[ActionWithTime]] = gather_pipeline_schedule(self.logger.rank_schedule.schedule, self.config.comm, log_window=log_window)

                # set the min_duration of each action block based on the lowerbound log time
                for ar_lb, actions_per_rank in zip(pipeline_schedule_lb, self.pipeline_schedule):
                    for a_lb, a in zip(ar_lb, actions_per_rank):
                        if a.freezable:
                            a.min_duration = a_lb.duration

                self.pipeline_schedule = set_freeze_ratio(self.pipeline_schedule, self.config)
                
                self.rand_noise_possibility = 0.1
                self.monitored_lb = True
                self.monitoring_lb = False
            elif self.monitored_lb:
                # TODO: after the warmup phase, freeze the model at the expected freeze ratio
                for a in self.pipeline_schedule[self.pp_rank]:
                    a.progressive_freezing = 1
                return
            else:
                return # Do nothing
        # Stable Freezing Phase - periodically adjust the freeze ratio
        elif curr_step % self.freeze_adjust_freq == 0:
            monitored_values_dict = {}
            for a, la in zip(self.pipeline_schedule[self.pp_rank], self.logger.rank_schedule):
                if not a.freezable:
                    continue
                times = la.log_time[self.progressive_freezing_start_step:]
                afrs = a.freeze_ratio_history[:len(times)]
                if a.stage not in monitored_values_dict.keys():
                    monitored_values_dict[a.stage] = []
                monitored_values_dict[a.stage] += [(afr, time) for (afr, time) in zip(afrs, times)] #  if (a.stage>0 or afr<=0.99)
            self.pipeline_schedule = adjust_freeze_ratio(self.pipeline_schedule, monitored_values_dict, self.config)
            logger.info(f"Adjusted Freeze Ratio per Block: {', '.join([f'[MB{a.microbatch}] {a.actual_freeze_ratio:.2f}/{a.expected_freeze_ratio:.2f}' for a in self.pipeline_schedule[self.pp_rank] if a.freezable])}")
            self.freeze_adjust_freq = self.freeze_adjust_freq * 2 # stop adjusting the freeze ratio
            self.rand_noise_possibility = self.rand_noise_possibility * 0.6
            return

    def _log_freeze_ratio(self):
        '''Update the frozen ratio based on the current freezing status.'''
        curr_step :int = self.logger.step_cnt
        if curr_step % self.stability_check_freq != 0:
            return

        for stage_idx, stage in self.stages.items():
            if self.monitoring_phase < curr_step:
                for name, _ in stage.named_parameters():
                    self.paramwise_frozen_count[stage_idx][name][0] = sum([a.paramwise_frozen_count[name][0] for a in self.pipeline_schedule[self.pp_rank] if a.stage == stage_idx and name in a.paramwise_frozen_count])
                    self.paramwise_frozen_count[stage_idx][name][1] = sum([a.paramwise_frozen_count[name][1] for a in self.pipeline_schedule[self.pp_rank] if a.stage == stage_idx and name in a.paramwise_frozen_count])

                average_freeze_ratio = float(np.mean([a.actual_freeze_ratio for a in self.pipeline_schedule[self.pp_rank] if a.stage == stage_idx and a.freezable]))
                self.freeze_ratio_history[stage_idx].append(average_freeze_ratio)
            else:
                self.freeze_ratio_history[stage_idx].append(0)
        
        if self.monitoring_phase < curr_step:
            # give a small noise to the expected freeze ratio until the next stability_check_freq
            # noise should be the same for all devices
            g = torch.Generator(device='cpu')
            g.manual_seed(curr_step)
            rand_noise = - int(torch.rand((), generator=g) < self.rand_noise_possibility) # no freezing policy with 2% possibility
            for action in self.pipeline_schedule[self.pp_rank]:
                action.rand_noise = rand_noise
        elif self.monitoring_lb:
            for action in self.pipeline_schedule[self.pp_rank]:
                action.rand_noise = 0
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
        self.stability_check_freq = 50 # frequency of stability check (paper: 50)
        self.alpha = 0.99 # parameter for exponential moving average (paper: 0.99)
        self.last_param = {name: None for stage in self.stages.values() for name, _ in stage.named_parameters()} # cumulative update
        self.ema = {name: 0.0 for stage in self.stages.values() for name, _ in stage.named_parameters()}
        self.ema_abs = {name: 0.0 for stage in self.stages.values() for name, _ in stage.named_parameters()}
        self.freezing_metric = {name: 0.0 for stage in self.stages.values() for name, _ in stage.named_parameters()} # effective perturbation metric. -> 0: oscillation (stable), -> 1: directional (unstable)

        # freeze update
        self.threshold = 0.05 # threshold on effective perturbation. (paper: 0.05)
        self.freezing_period = {name: 0 for stage in self.stages.values() for name, _ in stage.named_parameters()} # freeze the layer for a unit of stability_check_freq (paper: 50)
        self.frozen_due = {name: 0 for stage in self.stages.values() for name, _ in stage.named_parameters()}
        self.is_frozen = {name: False for stage in self.stages.values() for name, _ in stage.named_parameters()}
        return
    
    def freeze_update(self):
        curr_step :int = self.logger.step_cnt
        if not self.config.freezing.freeze or curr_step <= self.warmup_phase:
            return
        
        if curr_step % self.stability_check_freq == 0:
            self._calculate_freezing_metric()
            self._freeze_update() 
            self._update_freeze_ratio()

            last_efr_per_stage = {s: f'{self.freeze_ratio_history[s][-1]:.2f}' for s in self.freeze_ratio_history.keys()}
            logger.info(f"Freezing Ratio: {last_efr_per_stage}, Threshold: {self.threshold:.4e}")
        return
    
    def _calculate_freezing_metric(self):
        '''Calculate the effective perturbation metric for parameters.'''
        for stage in self.stages.values():
            for name, param in stage.named_parameters():
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
        for stage in self.stages.values():
            for name, param in stage.named_parameters():
                if not self.is_frozen[name]:
                    # update the freezing period (update in TCP style)
                    if self.freezing_metric[name] <= self.threshold:
                        self.freezing_period[name] += self.stability_check_freq
                else:
                    self.freezing_period[name] = self.freezing_period[name]/2
                    if self.freezing_period[name] < self.stability_check_freq:
                        self.freezing_period[name] = 0

                # update the freezing deadline
                self.frozen_due[name] = self.logger.step_cnt + self.freezing_period[name]

            # update the freezing status
            self.is_frozen[name] = self.logger.step_cnt < self.frozen_due[name]
            param.requires_grad = not self.is_frozen[name]
        return

class APFFreezerWithTimelyFreeze(_Freezer):
    def __init__(self, model_parts: List[Module], config: TimelyFreezeConfig):
        ''' Created on September 1, 2025
        Add LPSolver on top of the original APF.
        '''
        super().__init__(model_parts, config)

        self.progressive_freezing_phase = self.monitoring_phase + self.config.lr_scheduler.warmup_steps 
        '''Progressive Freezing Phase: gradually increase the freezing_params_num to the expected number.'''
        self.monitored_ub = False
        '''True if monitored the upperbound of batch time. set to True at the first stability check freq after monitoring phase.'''
        self.monitoring_lb = False 
        '''True if currently monitoring the lowerbound of batch time, i.e., freezing ratio = 1.0 for all actions.'''
        self.monitored_lb = False
        '''True if monitored the lowerbound of batch time, i.e., freezing ratio = 1.0 for all actions.'''
        self.monitoring_lb_start = None
        '''Starting step of monitoring lowerbound which is used in second monitoring phase.'''

        # Freezing Phases
        self.freeze_adjust_freq = self.phase_unit
        self.aggressiveness = self.config.freezing.aggressiveness # 0~1, default: 0.1
        self.progressive_freezing_start_step = None
        self.pipeline_schedule:List[List[ActionWithFreezing]] = [] # the pipeline schedule, which is a list of list of ActionWithFreezing
        
        # freezing metrics
        self.alpha = 0.99 # parameter for exponential moving average (paper: 0.99)
        self.last_param = {name: None for stage in self.stages.values() for name, _ in stage.named_parameters()} # cumulative update
        self.ema = {name: 0.0 for stage in self.stages.values() for name, _ in stage.named_parameters()}
        self.ema_abs = {name: 0.0 for stage in self.stages.values() for name, _ in stage.named_parameters()}
        self.freezing_metric = {name: 0.0 for stage in self.stages.values() for name, _ in stage.named_parameters()} # effective perturbation metric. -> 0: oscillation (stable), -> 1: directional (unstable)

        # freeze update
        self.threshold = 0.05 # threshold on effective perturbation. (paper: 0.05)
        self.freezing_period = {name: 0 for stage in self.stages.values() for name, _ in stage.named_parameters()} # freeze the layer for a unit of stability_check_freq (paper: 50)
        self.frozen_due = {name: 0 for stage in self.stages.values() for name, _ in stage.named_parameters()}
        self.freeze_cand = {stage_idx: {name: False for name, _ in stage.named_parameters()} for stage_idx, stage in self.stages.items()}
        self.freeze_ratio = {stage_idx: 0 for stage_idx in self.stages.keys()}
        return
    
    def freeze_update(self):
        curr_step :int = self.logger.step_cnt
        if not self.config.freezing.freeze or curr_step <= self.warmup_phase:
            return
        
        if curr_step % self.stability_check_freq == 0:
            self._log_freeze_ratio() # log the current (recent) freeze ratio
            self._set_expected_freeze_ratio() # update expected freeze ratio & random noise
            self._calculate_freezing_metric() # calculate freezing metric -> set actual list of freezing params
            
        # log the current and expected freeze ratio per microbatch block
        if curr_step % self.config.metrics.log_freq == 0 and self.monitoring_phase < curr_step and curr_step < self.progressive_freezing_phase + 2 * self.phase_unit:
            logger.info(f"Current/Expected Freeze Ratio per Block: {', '.join([f'[MB{a.microbatch}] {a.actual_freeze_ratio:.2f}/{a.expected_freeze_ratio:.2f}' for a in self.pipeline_schedule[self.pp_rank] if a.freezable])}")
        return
    
    
    def _calculate_freezing_metric(self):
        '''Calculate the effective perturbation metric for parameters, and set the actual list of freezing params.'''
        curr_step :int = self.logger.step_cnt
        for stage_idx, stage in self.stages.items():
            for i, (name, param) in enumerate(stage.named_parameters()):
                if not param.grad is None:
                    # calculate the effective perturbation metric
                    curr_param = param.detach().clone().cpu()  # current parameter value
                    if self.last_param[name] is None:
                        grad, grad_abs = 0.0, 0.0
                    else:
                        grad = (curr_param - self.last_param[name]).mean() # param.grad.mean().item()
                        grad_abs = (curr_param - self.last_param[name]).abs().mean() # param.grad.abs().mean().item()

                    self.last_param[name] = curr_param  # reset the cumulative update after calculating the metric
                    self.ema[name] = self.alpha * self.ema[name] + (1 - self.alpha) * grad
                    self.ema_abs[name] = self.alpha * self.ema_abs[name] + (1 - self.alpha) * grad_abs
                    freezing_metric = abs(self.ema[name]) / max(self.ema_abs[name], 1e-10)
                
                    # update the freezing period and due (update in TCP AIMD style)
                    if freezing_metric <= self.threshold:
                        self.freezing_period[name] += self.stability_check_freq
                    else:
                        self.freezing_period[name] = ((self.freezing_period[name]/2)//self.stability_check_freq)*self.stability_check_freq
                    self.frozen_due[name] = curr_step + self.freezing_period[name] # update the freezing deadline

                # update the freezing status
                self.freeze_cand[stage_idx][name] = curr_step < self.frozen_due[name]

        if curr_step <= self.monitoring_phase:
            return # do not freeze until Progressive Freezing Phase
        elif self.monitoring_lb:
            for action in self.pipeline_schedule[self.pp_rank]:
                action.rand_noise = 0
                action.freezing_list = [True] * action.num_params
                if action.stage == 0:
                    action.freezing_list[0] = False # do not freeze the first layer to consistently compute the input gradient
            return
        else:
            for action in self.pipeline_schedule[self.pp_rank]:
                freezable_num = action.num_params # - len(never_freeze[action.stage])
                freeze_cand_num = int(min(freezable_num, action.num_params*self.aggressiveness + sum([v for v in self.freeze_cand[action.stage].values()])))
                actual_num_freeze = min(freeze_cand_num, int(np.round(action.num_params * action.actual_freeze_ratio)))
                action.freezing_list = []
                if actual_num_freeze <= 0:
                    action.freezing_list = [False] * action.num_params
                else:
                    weights = [1 if val else 0.01 for val in self.freeze_cand[action.stage].values()]
                    # for name, val in self.freeze_cand[action.stage].items():
                    #     if val:
                    #         weights += [1.0] 
                    #     elif name in never_freeze[action.stage]:
                    #         weights += [0]
                    #     else:
                    #         weights += [0.01]
                    idx = torch.multinomial(torch.tensor(weights, dtype=torch.float16), actual_num_freeze, replacement=False)
                    freezing_list = torch.zeros(action.num_params, dtype=torch.bool)
                    freezing_list[idx] = True
                    action.freezing_list = freezing_list.tolist()
        return 


    def _set_expected_freeze_ratio(self):
        '''Set the expected freeze ratio based on the backward time.'''
        curr_step :int = self.logger.step_cnt
        # 0) Warmup + Monitoring Phase
        if curr_step <= self.monitoring_phase: 
            # during the monitoring phase, do not freeze the model
            # log_time(f"[Step {curr_step}] Monitoring Upperbound", master_only=True)
            self.monitored_ub = False
            return
        
        # 1) Monitor the upperbound of batch time
        elif not self.monitored_ub: # first stability check freq after monitoring phase
            logger.info(f"[Step {curr_step}] Setting Upperbound")

            # Set the stage module for each action in the pipeline schedule
            pipeline_schedule :List[List[ActionWithFreezing]] = set_freeze_ratio(gather_pipeline_schedule(self.logger.rank_schedule.schedule, self.config.comm), self.config)
            for a in pipeline_schedule[self.pp_rank]:
                a.module = self.stages[a.stage]
                a.freeze_flag = True
            self.pipeline_schedule = pipeline_schedule
            self.logger.action_dict = {(action.type, action.rank, action.microbatch, action.stage): action \
                                                for action in pipeline_schedule[self.pp_rank]}

            # Finish setting the upperbound of batch time
            self.monitored_ub = True
            self.progressive_freezing_start_step = curr_step

        # 2) Monitor the lowerbound of batch time
        elif not self.monitored_lb and not self.monitoring_lb:
            logger.info(f"[Step {curr_step}] Monitoring Lowerbound")
            for a in self.pipeline_schedule[self.pp_rank]:
                a.progressive_freezing = 1.0
                a.expected_freeze_ratio = 1.0 if a.stage > 0 else 1.0 - 1/a.num_params # don't freeze the first layer, to consistently maintain computing the input gradient

            self.monitoring_lb = True
            self.monitoring_lb_start = self.logger.step_cnt

        # 3) Set min_duration of each action block
        elif not self.monitored_lb and self.monitoring_lb:
            if curr_step <= self.monitoring_lb_start + 30 :
                return # wait for at least 30 steps
            else:
                logger.info(f"[Step {curr_step}] Setting Lowerbound")
                # create lowerbound pipeline schedule
                log_window = len(self.logger.rank_schedule[-1].log_time) - self.monitoring_lb_start
                pipeline_schedule_lb :List[List[ActionWithTime]] = gather_pipeline_schedule(self.logger.rank_schedule.schedule, self.config.comm, log_window=log_window)

                # set the min_duration of each action block based on the lowerbound log time
                for ar_lb, actions_per_rank in zip(pipeline_schedule_lb, self.pipeline_schedule):
                    for a_lb, a in zip(ar_lb, actions_per_rank):
                        if a.freezable:
                            a.min_duration = a_lb.duration
                
                # Set the expected freeze ratios
                self.pipeline_schedule = set_freeze_ratio(self.pipeline_schedule, self.config)
                
                self.monitored_lb = True
                self.monitoring_lb = False

        # 4) Stable Freezing Phase - periodically adjust the freeze ratio 
        elif curr_step % self.freeze_adjust_freq == 0: 
            monitored_values_dict = {a.stage: [] for a in self.pipeline_schedule[self.pp_rank] if a.freezable}
            for a, la in zip(self.pipeline_schedule[self.pp_rank], self.logger.rank_schedule):
                if not a.freezable:
                    continue
                times = la.log_time[self.progressive_freezing_start_step:]
                afrs = a.freeze_ratio_history[:len(times)]
                monitored_values_dict[a.stage] += [(afr, time) for (afr, time) in zip(afrs, times)] 
            self.pipeline_schedule = adjust_freeze_ratio(self.pipeline_schedule, monitored_values_dict, self.config)
            logger.info(f"Adjusted Freeze Ratio per Block: {', '.join([f'[MB{action.microbatch}] {action.actual_freeze_ratio:.2f}/{action.expected_freeze_ratio:.2f}' for action in self.pipeline_schedule[self.pp_rank] if action.freezable])}")
            self.freeze_adjust_freq = self.freeze_adjust_freq * 2 # stop adjusting the freeze ratio
            self.aggressiveness = min(self.aggressiveness * 2, 1.0)
            return

    def _log_freeze_ratio(self):
        '''Log the previous frozen ratio based on the freezing status.'''
        curr_step :int = self.logger.step_cnt
        if curr_step % self.stability_check_freq != 0:
            return

        for stage_idx, stage in self.stages.items(): 
            if self.monitoring_phase < curr_step and len(self.pipeline_schedule) > 0:
                for name, _ in stage.named_parameters():
                    self.paramwise_frozen_count[stage_idx][name][0] = sum([a.paramwise_frozen_count[name][0] for a in self.pipeline_schedule[self.pp_rank] if a.stage == stage_idx and name in a.paramwise_frozen_count])
                    self.paramwise_frozen_count[stage_idx][name][1] = sum([a.paramwise_frozen_count[name][1] for a in self.pipeline_schedule[self.pp_rank] if a.stage == stage_idx and name in a.paramwise_frozen_count])

                average_freeze_ratio = float(np.mean([a.actual_freeze_ratio for a in self.pipeline_schedule[self.pp_rank] if a.stage == stage_idx and a.freezable]))
                self.freeze_ratio_history[stage_idx].append(average_freeze_ratio)
            else:
                self.freeze_ratio_history[stage_idx].append(0)
        return

