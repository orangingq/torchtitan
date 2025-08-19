import os
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Literal, Optional
from torch.distributed import ProcessGroup
import torch.distributed as dist

from torchtitan.config.job_config import JobConfig, \
                    Training as BaseTraining, \
                    Metrics as BaseMetrics, \
                    Comm as BaseComm, \
                    Parallelism as BaseParallelism

@dataclass
class Training(BaseTraining):
    pass

@dataclass
class Metrics(BaseMetrics):
    log_freq: int = 100
    """Frequency of logging during training (steps)"""

    wandb_name: str | None = None
    """Weights & Biases run name"""

    basename: str = time.strftime('%y%m%d_%H%M')
    """Base name for logs, image saving, etc. Initially, it will be set to the current time."""
    
    draw_graph: bool = False
    """Whether to draw graphs of the model and training process. Can lower throughput due to graph generation overhead."""


@dataclass
class PipelineParallelism(BaseParallelism):
    pp_scheduler: Literal["GPipe", "1FB", "Interleaved1F1B", "InterleavedZeroBubble", "ZBVZeroBubble"] | None = None
    """Pipeline parallelism scheduler. Options: 'gpipe', '1F1B', 'Interleaved1F1B', 'InterleavedZeroBubble' 'ZBVZeroBubble'."""

    stages_per_rank: int = 1 
    """Number of pipeline stages per rank. >1 for interleaved pipeline parallelism, 1 for non-interleaved pipeline parallelism."""

    num_stages: int = 1 
    """Number of pipeline parallelism stages. It will be recalculated later as: pp * stages_per_rank."""

    stages_list : list[int] = None
    """List of pipeline stages of this rank. It will be recalculated based on the pipeline schedule."""

    @property
    def bwd_separated(self) -> bool:
        """
        Check if the backward pass is separated into backward_weight and backward_input.
        Returns True if the pp_scheduler is 'InterleavedZeroBubble', or 'ZBVZeroBubble'.
        """
        return self.pp_scheduler in ["InterleavedZeroBubble", "ZBVZeroBubble"]

    microbatches: int = 4 # number of microbatches for pipeline parallelism
    """Number of microbatches for pipeline parallelism."""

@dataclass
class Freezing:
    """Configuration for model freezing during training.
    This is used to freeze the model weights based on a specific metric.
    """
    freeze: bool = False
    """Whether to enable TimelyFreeze."""

    metric_type: str | None = None
    """Metric type for freezing."""
    
    phase_unit: int = 100
    """Number of steps per phase for freezing."""


@dataclass
class Comm(BaseComm):
    """
    Communication configuration for distributed training.
    """

    world_size: int = 1
    """Total number of processes participating in the job."""

    global_rank: int = 0
    """Rank of the current process in the whole (multi-node) distributed system."""

    local_rank: int = 0
    """Local rank of the current process within its node."""

    dp: int = 1 
    """Data parallelism degree (number of data parallelism units)."""

    pp: int = 1
    """Pipeline parallelism degree (number of pipeline parallelism units)."""

    pp_group: Optional[ProcessGroup] = None
    """Pipeline parallelism group (having same dp_rank, but different pp_rank) of this process."""

    dp_group: Optional[ProcessGroup] = None
    """Data parallelism group (having same pp_rank, but different dp_rank) of this process."""

    pp_rank: int = 0  # pipeline parallelism rank : local rank in the pipeline parallelism group
    """pp_rank" is the local rank in the pipeline parallelism group."""
    
    dp_rank: int = 0  # data parallelism rank : local rank in the data parallelism group
    """dp_rank" is the local rank in the data parallelism group."""

    # master_rank: int = dp * (pp - 1)
    # """master_rank" is the global rank of the master process that manages the whole training processes."""

    master_dp_rank: int = 0  
    """master_dp_rank" is the local rank in the data parallelism group of the master process."""

    @property
    def distributed(self) -> bool:
        """
        Whether the job is distributed (i.e., uses multiple processes).
        Returns True if either data parallelism or pipeline parallelism is enabled.
        """
        return self.world_size > 1
    
    @property
    def get_first_stage_rank(self) -> int:
        '''get the first stage rank in the pipeline parallelism group.'''
        return self.dp_rank # args.dp_rank * args.pp 
    
    @property
    def is_first_stage(self) -> bool:
        '''check if this process is the first stage in the pipeline parallelism group.'''
        return self.global_rank == self.get_first_stage_rank

    @property
    def get_last_stage_rank(self) -> int:
        '''get the last stage rank in the pipeline parallelism group.'''
        return self.dp_rank + self.dp * (self.pp - 1) # args.dp_rank * args.pp + (args.pp - 1)
    
    @property
    def is_last_stage(self) -> bool:
        '''check if this process is the last stage in the pipeline parallelism group.'''
        return self.global_rank == self.get_last_stage_rank

    @property
    def is_master_rank(self) -> bool:
        '''check if this process is the master rank.'''
        return self.global_rank == self.dp * (self.pp - 1) # self.master_rank


@dataclass
class TimelyFreezeConfig(JobConfig):
    """
    Default container for training configuration.
    """
    training: Training = field(default_factory=Training)
    metrics: Metrics = field(default_factory=Metrics)
    parallelism: PipelineParallelism = field(default_factory=PipelineParallelism)
    freezing: Freezing = field(default_factory=Freezing)
    comm: Comm = field(default_factory=Comm)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def print(self) -> None:
        """
        Print the configuration in a readable format.
        """
        print("----- TimelyFreeze Configuration:")
        for key, value in self.to_dict().items():
            if isinstance(value, dict):
                print(f"{key}:")
                for sub_key, sub_value in value.items():
                    print(f"  {sub_key}: {sub_value}")
            else:
                print(f"{key}: {value}")
        return

    def update_config(self, trainer) -> None:
        """ Update the TimelyFreezeConfig from a parallel_dim object.
        This is useful for integrating with existing Trainer configurations.
        """
        # Update log file/folder names
        if self.metrics.wandb_name is None:
            self.metrics.wandb_name = self.metrics.basename
        self.checkpoint.folder = os.path.join(self.checkpoint.folder, self.metrics.basename)
        self.comm.save_traces_folder = os.path.join(self.comm.save_traces_folder, self.metrics.basename)
        self.profiling.save_traces_folder = os.path.join(self.profiling.save_traces_folder, self.metrics.basename)
        self.profiling.save_memory_snapshot_folder = os.path.join(self.profiling.save_memory_snapshot_folder, self.metrics.basename)
        self.metrics.save_tb_folder = os.path.join(self.metrics.save_tb_folder, self.metrics.basename)

        # Update Parallelism configs
        self.parallelism.stages_per_rank = 1 if self.parallelism.pipeline_parallel_schedule.lower() in ["gpipe", "1f1b"] else 2
        self.parallelism.num_stages = len(trainer.model_parts)
        self.parallelism.microbatches = trainer.pp_schedule._n_microbatches # self.training.local_batch_size // self.parallelism.pipeline_parallel_microbatch_size
        self.parallelism.stages_list = list(set(a.stage_index for a in trainer.pp_schedule.pipeline_order[self.comm.local_rank] if a is not None))

        self.comm.world_size = trainer.parallel_dims.world_size
        self.comm.global_rank = trainer.parallel_dims.world_mesh.get_local_rank() #TODO
        self.comm.local_rank = trainer.parallel_dims.world_mesh.get_local_rank()
        
        # Data Parallelism
        if trainer.parallel_dims.dp_enabled:
            dp_mesh = trainer.parallel_dims.world_mesh["dp"]
            self.comm.dp, self.comm.dp_rank = dp_mesh.size(), dp_mesh.get_local_rank()
            self.comm.dp_group = dp_mesh.get_group("dp")
        else:
            self.comm.dp, self.comm.dp_rank = 1, 0
            self.comm.dp_group = dist.group.WORLD
        
        # Pipeline Parallelism
        if trainer.parallel_dims.pp_enabled:
            pp_mesh = trainer.parallel_dims.world_mesh["pp"]
            self.comm.pp, self.comm.pp_rank = pp_mesh.size(), pp_mesh.get_local_rank()
            self.comm.pp_group = pp_mesh.get_group("pp")
        else:
            self.comm.pp, self.comm.pp_rank = 1, 0
            self.comm.pp_group = dist.group.WORLD

        return
    
global_config = TimelyFreezeConfig()