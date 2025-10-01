from torch.cuda import nvtx as cuda_nvtx
import nvtx
from .config import global_config

def mark_step_epoch_nvtx(step, epoch=0):
    '''print the epoch nvtx'''
    nvtx.mark(message=f"step {step} of epoch {epoch}")
    return

def mark_nvtx(message, step=None):
    '''print the update nvtx'''
    if step is None:
        nvtx.mark(message=f"{message}", color="cyan", domain=f"Rank{global_config.comm.pp_rank}")
    else:
        nvtx.mark(message=f"{message} of step {step}", color="cyan", domain=f"Rank{global_config.comm.pp_rank}")
    return

def forward_nvtx(microbatch, start=True):
    '''mark the range of forward in nvtx'''
    if start:
        nvtx.push_range(f"{microbatch}_forward", color="blue", domain=f"Rank{global_config.comm.pp_rank}", category="forward")
        cuda_nvtx.range_push(f"{microbatch}_forward")
    else:
        cuda_nvtx.range_pop()
        nvtx.pop_range(f"Rank{global_config.comm.pp_rank}")
    return

def backward_nvtx(microbatch, start=True):
    '''mark the range of backward in nvtx'''
    if start:
        nvtx.push_range(f"{microbatch}_backward", color="green", domain=f"Rank{global_config.comm.pp_rank}", category="backward")
        cuda_nvtx.range_push(f"{microbatch}_backward")
    else:
        cuda_nvtx.range_pop()
        nvtx.pop_range(f"Rank{global_config.comm.pp_rank}")
    return

def backward_input_nvtx(microbatch, start=True):
    '''mark the range of backward in nvtx'''
    if start:
        nvtx.push_range(f"{microbatch}_backward_input", color="#08CEC5", domain=f"Rank{global_config.comm.pp_rank}", category="backward_input")
        cuda_nvtx.range_push(f"{microbatch}_backward_input")
    else:
        cuda_nvtx.range_pop()
        nvtx.pop_range(f"Rank{global_config.comm.pp_rank}")
    return


def comm_nvtx(message, microbatch, start=True):
    '''mark the range of forward/backward send/recv period in nvtx'''
    if start:
        nvtx.push_range(f"{microbatch}_{message}", color="gray", domain=f"Rank{global_config.comm.pp_rank}_comm")
    else:
        nvtx.pop_range(f"Rank{global_config.comm.pp_rank}_comm")
    return


def grad_sync_nvtx(microbatch, start=True):
    '''mark the range of gradient synchronization in nvtx'''
    if start:
        nvtx.push_range(f"{microbatch}_grad_sync", color="yellow", domain=f"Rank{global_config.comm.pp_rank}", category="grad_sync")
    else:
        nvtx.pop_range(f"Rank{global_config.comm.pp_rank}")
    return