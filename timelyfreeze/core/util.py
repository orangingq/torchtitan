import os
from datetime import datetime
from typing import List
from matplotlib import pyplot as plt
from scipy import stats
import numpy as np

from .action import ActionType, ActionWithTime
from .config import global_config 

default_path = {
    "checkpoint": "/data2/shcho/torchtitan/checkpoints",
    "dataset": "/data2/shcho/datasets",
    "wandb": "/data2/shcho/torchtitan/wandb", 
    "log": "/home/shcho/torchtitan/logs",
    "image": "/data2/shcho/torchtitan/images"
}

def get_default_path(key:str)->str:
    '''Get the default path of the specified key. 
    Args:
        key (str): the key of the default path. 
    '''
    assert key in default_path.keys(), f"key must be one of {default_path.keys()}"
    return default_path[key]

def get_abs_path(path:str, key:str, make=False)->str:
    '''Get the absolute path of the specified path. 
    Args:
        path (str): the path to be joined with the default path.
        key (str): the key of the default path. 
    '''
    assert key in default_path.keys(), f"key must be one of {default_path.keys()}"
    
    if path is None:
        return get_default_path(key)
    if not os.path.isabs(path):
        path = os.path.abspath(os.path.join(default_path[key], path))
    if make:
        os.makedirs(os.path.dirname(path), exist_ok=True)
    return path


def log_time(message='', rank=True, timestamp=None, end='\n', master_only=False):
    '''log the time.'''
    if master_only and not global_config.comm.is_master_rank:
        return message
    if timestamp is None:
        timestamp = datetime.now().strftime('%y-%m-%d %H:%M:%S.%f')[:-2] + ' '
    rank = f"[RANK{global_config.comm.global_rank}] " if rank else ''
    message = f"{timestamp}{rank}{message}{end}"
    print(message, end='', flush=True)
    return message


def draw_line_chart(data_x, data_y, save_file, title=None, xlabel=None, ylabel=None):
    '''Draw the line chart of the data.'''
    if len(data_x) == 0 or len(data_y) == 0:
        log_time("Data is empty. Skip drawing the line chart.")
        return None
    elif len(data_x) != len(data_y):
        log_time(f"The length of data_x [{len(data_x)}] and data_y [{len(data_y)}] should be the same.")
        return None
    
    fig, ax = plt.subplots()
    # Fit and plot trend line
    if len(data_y) > 30:
        window_size = 5
        moving_avg = np.convolve(data_y, np.ones(window_size)/window_size, mode='valid')
        ax.plot(data_x[window_size-1:], moving_avg, linestyle='-', color='#EDE8DF', linewidth=10)
    ax.plot(data_x, data_y, marker='o', linestyle='-', color='#988456', markersize=2 if len(data_y) > 50 else 3, linewidth=1)
    ax.set_title(title if title is not None else f"Line Chart of Rank {global_config.comm.global_rank} (Stage {global_config.comm.pp_rank})", 
                    fontdict={'fontsize': 11})
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    save_file = get_abs_path(save_file, 'image', make=True)
    plt.savefig(save_file)
    plt.close()
    log_time(f"{title if title is not None else 'Line Chart'}  is saved as: {save_file}")
    return save_file

def draw_time_histogram(time_list, save_file, title=None, xlabel=None, ylabel=None, beta=False):
    '''Draw the time histogram of the model.'''
    time_list = np.array(time_list)

    if len(time_list) < 10:
        log_time(f"time_list is too few (length={len(time_list)}). Skip drawing the histogram.")
        return None

    min_x = np.quantile(time_list, 0.05) # if np.quantile(time_list, 0.01) > 0 else min(np.quantile(time_list, 0.01), 5)
    max_x = np.quantile(time_list, 0.95) # if np.quantile(time_list, 0.99) < 10 else max(np.quantile(time_list, 0.99), 20)
    time_list = time_list[(time_list > min_x) & (time_list < max_x)]
    
    if len(time_list) == 0:
        log_time("time_list is too few. Skip drawing the histogram.")
        return None

    # Plot histogram
    fig, ax1 = plt.subplots()
    ax1.set_xlabel(f'{xlabel if xlabel else "Time (ms)"}')
    ax1.set_ylabel(f'{ylabel if ylabel else "Frequency"}')
    
    mean_time, std_time = np.mean(time_list), np.std(time_list)
    counts, bin_edges, _ = ax1.hist(time_list, bins=50, color='#AE9B71')
    peak_x, peak_y = np.mean(bin_edges[np.argmax(counts):np.argmax(counts)+2]), np.max(counts)
    ax1.set_ylim(0, ax1.get_ylim()[1])

    # Generate beta distribution
    if beta:
        time_list_beta = (time_list - min_x) / (max_x - min_x)
        alpha, beta = calculate_beta_distribution(time_list_beta)
        x = np.linspace(0, 1, 100)
        ax2 = ax1.twinx()
        ax2.plot(x * (max_x-min_x) + min_x, stats.beta.pdf(x, alpha, beta), '-', lw=2, 
                label=f'Beta Distribution (a={alpha:.2f}, b={beta:.2f})', color='#BEB7B1')
        ax2.axvline(x=mean_time, label=f'Mean: {mean_time:.2f} ms, Std. Dev.: {std_time:.2f} ms', color='#BEB7B1', linestyle='--', linewidth=1)
        ax2.plot(peak_x, peak_y/(ax1.get_ylim()[1].item())*ax2.get_ylim()[1], label=f'Peak: {peak_x:.2f} ms', color='#433B26', marker='o', markersize=2)
        ax2.set_ylabel('PDF')
        ax2.tick_params(axis='y', which='both', direction='in', left=False, right=True)
        ax2.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
        ax2.set_ylim(0, ax2.get_ylim()[1])
    else:
        ax1.axvline(x=mean_time, label=f'Mean: {mean_time:.2f} ms, Std. Dev.: {std_time:.2f} ms', color='#BEB7B1', linestyle='--', linewidth=1)
        ax1.plot(peak_x, peak_y, label=f'Peak: {peak_x:.2f} ms', color='#433B26', marker='o', markersize=2)
    
    plt.legend()
    plt.xlim(min_x, max_x)
    plt.title(title if title is not None else f"Time Histogram of Rank {global_config.comm.global_rank} (Stage {global_config.comm.pp_rank})", fontsize=11)
    save_file = get_abs_path(save_file, 'image', make=True)
    plt.savefig(save_file)
    plt.close()
    log_time(f'{title if title is not None else "Time Histogram"} is saved as: {save_file}\n\t> Length: {len(time_list)}, Mean: {mean_time:.2f} ms, Std. Dev.: {std_time:.2f} ms')

    return save_file

def calculate_beta_distribution(datapoints):
    '''Calculate alpha and beta for the beta distribution.'''
    mean = np.mean(datapoints)
    sigma = np.std(datapoints)
    var = sigma ** 2
    alpha = ((1 - mean) / var - 1 / mean) * mean ** 2
    beta = alpha * (1 / mean - 1)
    return alpha, beta

def draw_elementwise_histogram(data, stage, save_file, title=None, xlabel1=None, xlabel2=None):
    '''
    Draw the elementwise histogram of the model.
    data: list of the count and total per element. [[count: int, total: int] for each element].    
    '''
    total_sum = sum([data_l[1] for data_l in data])
    if total_sum == 0:
        log_time("data is empty. Skip drawing the elementwise histogram.")
        return None
    
    fig, (ax1, ax2) = plt.subplots(2, 1, 
        figsize=(10, 4) if len(data) < 20 else (15, 4),  
        gridspec_kw={'height_ratios': [1, 3]}  # Relative heights: 3:1
    )
    bar_width = 0.8
    bar_colors = np.array([
        [246, 244, 239],  # #f6f4ef
        [67, 59, 38]  # #433b26
    ]) / 255.0
    bar_colors = np.linspace(bar_colors[0], bar_colors[1], len(data) * (global_config.parallelism.num_stages + 1))
    bar_colors = bar_colors[stage * len(data):(stage + 1) * len(data)]

    past_counts = 0
    for i, (data_l, color) in enumerate(zip(data, bar_colors)):
        # subgraph 1 showing the (count/total) of each element
        count, total = data_l
        ratio = count/total_sum * 100
        past_ratio = past_counts/total_sum * 100
        ax1.barh(0, ratio, height=bar_width/3, color=color, edgecolor='none', left=past_ratio)
        
        # subgraph 2 showing the count per element
        ratio_per_element = count / total
        ax2.bar(i, ratio_per_element, color=color, edgecolor='none')
        
        past_counts += count
    
    total_count_text = f'Counts Sum\n{int(past_counts)}\n({past_counts/total_sum*100:.2f}%)'
    ax1.text(max(min(80, past_ratio-5),5), 0, total_count_text, ha='center', va='center', fontsize=10, color='black')
    total_elem_text = f'Total Sum\n{int(total_sum)}\n(100%)'
    ax1.text(95, 0, total_elem_text, ha='center', va='center', fontsize=10, color='black')
    ax1.set_xlim(0, 100)
    ax1.set_xticks(np.arange(0, 101, 10))
    ax1.set_yticks([])
    ax1.set_xlabel(f'{xlabel1 if xlabel1 else "Total Counts"} (%)')
    plt.subplots_adjust(hspace=0.6)

    ax2.set_xlim(-0.5, len(data) - 0.5)
    ax2.set_ylim(0, 1)
    xticks = range(len(data)) if len(data) < 20 else range(0, len(data), len(data)//20)
    ax2.set_xticks(xticks)
    ax2.set_xticklabels([f'{i}' for i in xticks], rotation=45, fontsize=8)
    ax2.set_ylabel('Count/Total Ratio')
    ax2.set_xlabel(f'{xlabel2 if xlabel2 else "Element Index"} (#)')

    fig.suptitle(title if title is not None else f"Elementwise Histogram of Rank {global_config.comm.global_rank} (Stage {stage})", fontsize=13)
    plt.subplots_adjust(bottom=0.2)

    save_file = get_abs_path(save_file, 'image', make=True)
    plt.savefig(save_file)
    plt.close()
    log_time(f"Elementwise Histogram is saved as: {save_file}\n\t> Counts Sum: {int(past_counts)}, Total Sum: {int(total_sum)} ({past_counts/total_sum*100:.2f}%), " \
             + f"Ratio(%) per element: {[int(data_l[0] / data_l[1]*10000)/100 for data_l in data]}")
    return


def draw_pipeline_schedule(save_file:str, 
                            pipeline_schedule:List[List[ActionWithTime]],
                            title=None, xlabel=None, ylabel=None):
    num_ranks = len(pipeline_schedule) # same as global_config.parallelism.pp
    stages_per_rank = [list(dict.fromkeys([action.stage for action in pipeline_schedule[rank] if action.type == ActionType.FORWARD and action.stage is not None])) for rank in range(num_ranks)]
    num_stages_per_rank = len(stages_per_rank[0]) # same as global_config.parallelism.stages_per_rank

    # draw the pipeline schedule
    max_time = max([actions_per_rank[-1].end_time for actions_per_rank in pipeline_schedule])    
    space = 0 # max_time / 100
    color_map = {
        ActionType.FORWARD: '#4285F4', # Blue
        ActionType.FULL_BACKWARD: '#34A853', # Green
        ActionType.BACKWARD_INPUT: '#46BDC6', # Green + blue
        ActionType.BACKWARD_WEIGHT: '#34A853',
    }
    stage_color_map = [{stage: f'#{int(255-s/(num_stages_per_rank-0.999)*255):02X}{int(255-s/(num_stages_per_rank-0.999)*255):02X}{int(255-s/(num_stages_per_rank-0.999)*255):02X}' 
                        for s, stage in enumerate(stages_per_rank[rank])} for rank in range(num_ranks)]

    # set the figure size and axes
    fig, ax = plt.subplots(figsize=(max(1, round(max_time/(50 if max_time < 400 else 100)*3)), 3), dpi=100)
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.margins(0)
    if not (xlabel or ylabel): 
        ax.axis('off')
    ax.spines['bottom'].set_position(('outward', 0))
    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.set_xlim(-space, max_time + space) # set x-axis limit
    ax.invert_yaxis() # invert y-axis to have rank 0 at the top
    ax.set_xticks(np.append(np.arange(0, max_time if (max_time%100>=30) else max_time-100, 100), max_time))
    ax.set_yticks(range(num_ranks))
    ax.set_yticklabels([f'Rank {i}' for i in range(num_ranks)])
    
    if xlabel:
        ax.set_xlabel(xlabel) # "Time"
    if ylabel:
        ax.set_ylabel(ylabel) # "Rank"
    if title is not None:
        ax.set_title(title) # "Pipeline Schedule TimeBlock Sequence" if title is None else title
    
    # Add vertical construction lines for better visualization
    for time in plt.xticks()[0]: 
        ax.axvline(x=time, color='gray', linestyle='--', linewidth=0.5, alpha=0.7, zorder=0)

    # Add bubble ratio text on the right side of the plot
    up_time = max([actions_per_rank[-1].end_time - actions_per_rank[0]._start_time for actions_per_rank in pipeline_schedule])
    gpu_util_time = [sum([action.duration for action in actions_per_rank]) for actions_per_rank in pipeline_schedule]
    gpu_bubble_ratio = [(1- gpu_util_time[rank] / up_time) for rank in range(num_ranks)]

    # draw the pipeline schedule blocks 
    for actions_per_rank in pipeline_schedule:
        for action in actions_per_rank:
            # draw the action block
            if global_config.freezing.bwd_separated and action.type == ActionType.FULL_BACKWARD:
                w_i = action.min_duration if hasattr(action, 'min_duration') else action.duration/2
                ax.barh(
                    y=action.rank, width=w_i, left=action._start_time,
                    height=1, color=color_map[ActionType.BACKWARD_INPUT], edgecolor='white', label=action.microbatch)
                ax.barh( 
                    y=action.rank, width=action.duration - w_i, left=action._start_time+w_i,
                    height=1, color=color_map[ActionType.BACKWARD_WEIGHT], edgecolor='white', label=action.microbatch)
                # write the microbatch index in the middle of the block
                if action.duration > 1: # only write the microbatch index if the duration is greater than 1 ms
                    ax.text(
                        x=action._start_time + w_i/2, y=action.rank, s=str(action.microbatch+1), # +1 for 1-based index
                        ha='center', va='center', fontsize=15,
                        color=stage_color_map[action.rank].get(action.stage, 'black')
                    )
                    ax.text(
                        x=action._start_time + (action.duration + w_i)/2, y=action.rank, s=str(action.microbatch+1), # +1 for 1-based index
                        ha='center', va='center', fontsize=15,
                        color=stage_color_map[action.rank].get(action.stage, 'black')
                    )
            else:
                ax.barh( 
                    y=action.rank, width=action.duration, left=action._start_time,
                    height=1, color=color_map[action.type], edgecolor='white', label=action.microbatch)
                # write the microbatch index in the middle of the block
                if action.duration > 1: # only write the microbatch index if the duration is greater than 1 ms
                    ax.text(
                        x=action._start_time + action.duration/2, y=action.rank, s=str(action.microbatch+1), # +1 for 1-based index
                        ha='center', va='center', fontsize=15,
                        color=stage_color_map[action.rank].get(action.stage, 'black')
                    )
    
    save_file = get_abs_path(save_file, 'image', make=True)
    plt.savefig(save_file, bbox_inches='tight', pad_inches=0)
    plt.close()
    log_time(f"Pipeline schedule is saved as: {save_file}\n\t> Batch Time: {up_time:.2f} ms, GPU Bubble Ratio: {', '.join([f'{val*100:.2f}%' for val in gpu_bubble_ratio])}")

    return

if __name__ == "__main__":
    # Test the draw_pipeline_schedule function
    from .schedule import ActionType, ActionWithTime, link_actions, schedule_pipeline

    # Example pipeline schedule 1 : easy example
    pipeline_schedule = [
        [ActionWithTime(ActionType.FORWARD, 0, 0), ActionWithTime(ActionType.FULL_BACKWARD, 0, 0)],
        [ActionWithTime(ActionType.FORWARD, 1, 0), ActionWithTime(ActionType.FULL_BACKWARD, 1, 0)],
        [ActionWithTime(ActionType.FORWARD, 2, 0), ActionWithTime(ActionType.FULL_BACKWARD, 2, 0)]
    ]
    fwd_time = [1.0, 1.5, 2.0]
    bwd_time = [1.5, 2.0, 2.5]
    pipeline_schedule = schedule_pipeline(link_actions(pipeline_schedule), fwd_time, bwd_time)
    draw_pipeline_schedule("pipeline_schedule.pdf", pipeline_schedule)

    # Example pipeline schedule 2 : interleavedzb, 4 ranks, 8 microbatches, 2 stages per rank
    # 2-1 : Realistic version
    pipeline_schedule = [
        [ActionWithTime(1, 0, 0, 0), ActionWithTime(1, 0, 1, 0), ActionWithTime(1, 0, 2, 0), ActionWithTime(1, 0, 3, 0), ActionWithTime(1, 0, 0, 4), ActionWithTime(1, 0, 1, 4), ActionWithTime(1, 0, 2, 4), ActionWithTime(1, 0, 3, 4), ActionWithTime(2, 0, 0, 4), ActionWithTime(3, 0, 0, 4), ActionWithTime(1, 0, 4, 0), ActionWithTime(2, 0, 1, 4), ActionWithTime(3, 0, 1, 4), ActionWithTime(1, 0, 5, 0), ActionWithTime(2, 0, 2, 4), ActionWithTime(3, 0, 2, 4), ActionWithTime(1, 0, 6, 0), ActionWithTime(2, 0, 3, 4), ActionWithTime(3, 0, 3, 4), ActionWithTime(1, 0, 7, 0), ActionWithTime(10, 0, 0, 0), ActionWithTime(1, 0, 4, 4), ActionWithTime(10, 0, 1, 0), ActionWithTime(1, 0, 5, 4), ActionWithTime(10, 0, 2, 0), ActionWithTime(1, 0, 6, 4), ActionWithTime(10, 0, 3, 0), ActionWithTime(1, 0, 7, 4), ActionWithTime(2, 0, 4, 4), ActionWithTime(3, 0, 4, 4), ActionWithTime(2, 0, 5, 4), ActionWithTime(3, 0, 5, 4), ActionWithTime(2, 0, 6, 4), ActionWithTime(3, 0, 6, 4), ActionWithTime(2, 0, 7, 4), ActionWithTime(3, 0, 7, 4), ActionWithTime(10, 0, 4, 0), ActionWithTime(10, 0, 5, 0), ActionWithTime(10, 0, 6, 0), ActionWithTime(10, 0, 7, 0)],
        [ActionWithTime(1, 1, 0, 1), ActionWithTime(1, 1, 1, 1), ActionWithTime(1, 1, 2, 1), ActionWithTime(1, 1, 3, 1), ActionWithTime(1, 1, 0, 5), ActionWithTime(1, 1, 1, 5), ActionWithTime(1, 1, 2, 5), ActionWithTime(2, 1, 0, 5), ActionWithTime(1, 1, 3, 5), ActionWithTime(2, 1, 1, 5), ActionWithTime(3, 1, 0, 5), ActionWithTime(1, 1, 4, 1), ActionWithTime(2, 1, 2, 5), ActionWithTime(3, 1, 1, 5), ActionWithTime(1, 1, 5, 1), ActionWithTime(2, 1, 3, 5), ActionWithTime(3, 1, 2, 5), ActionWithTime(1, 1, 6, 1), ActionWithTime(2, 1, 0, 1), ActionWithTime(3, 1, 3, 5), ActionWithTime(1, 1, 7, 1), ActionWithTime(2, 1, 1, 1), ActionWithTime(3, 1, 0, 1), ActionWithTime(1, 1, 4, 5), ActionWithTime(2, 1, 2, 1), ActionWithTime(3, 1, 1, 1), ActionWithTime(1, 1, 5, 5), ActionWithTime(2, 1, 3, 1), ActionWithTime(3, 1, 2, 1), ActionWithTime(1, 1, 6, 5), ActionWithTime(2, 1, 4, 5), ActionWithTime(3, 1, 3, 1), ActionWithTime(1, 1, 7, 5), ActionWithTime(2, 1, 5, 5), ActionWithTime(3, 1, 4, 5), ActionWithTime(2, 1, 6, 5), ActionWithTime(3, 1, 5, 5), ActionWithTime(2, 1, 7, 5), ActionWithTime(3, 1, 6, 5), ActionWithTime(2, 1, 4, 1), ActionWithTime(3, 1, 7, 5), ActionWithTime(2, 1, 5, 1), ActionWithTime(3, 1, 4, 1), ActionWithTime(2, 1, 6, 1), ActionWithTime(3, 1, 5, 1), ActionWithTime(2, 1, 7, 1), ActionWithTime(3, 1, 6, 1), ActionWithTime(3, 1, 7, 1)], 
        [ActionWithTime(1, 2, 0, 2), ActionWithTime(1, 2, 1, 2), ActionWithTime(1, 2, 2, 2), ActionWithTime(1, 2, 3, 2), ActionWithTime(1, 2, 0, 6), ActionWithTime(1, 2, 1, 6), ActionWithTime(2, 2, 0, 6), ActionWithTime(1, 2, 2, 6), ActionWithTime(2, 2, 1, 6), ActionWithTime(1, 2, 3, 6), ActionWithTime(2, 2, 2, 6), ActionWithTime(3, 2, 0, 6), ActionWithTime(1, 2, 4, 2), ActionWithTime(2, 2, 3, 6), ActionWithTime(3, 2, 1, 6), ActionWithTime(1, 2, 5, 2), ActionWithTime(2, 2, 0, 2), ActionWithTime(3, 2, 2, 6), ActionWithTime(1, 2, 6, 2), ActionWithTime(2, 2, 1, 2), ActionWithTime(3, 2, 3, 6), ActionWithTime(1, 2, 7, 2), ActionWithTime(2, 2, 2, 2), ActionWithTime(3, 2, 0, 2), ActionWithTime(1, 2, 4, 6), ActionWithTime(2, 2, 3, 2), ActionWithTime(3, 2, 1, 2), ActionWithTime(1, 2, 5, 6), ActionWithTime(2, 2, 4, 6), ActionWithTime(3, 2, 2, 2), ActionWithTime(1, 2, 6, 6), ActionWithTime(2, 2, 5, 6), ActionWithTime(3, 2, 3, 2), ActionWithTime(1, 2, 7, 6), ActionWithTime(2, 2, 6, 6), ActionWithTime(3, 2, 4, 6), ActionWithTime(2, 2, 7, 6), ActionWithTime(3, 2, 5, 6), ActionWithTime(2, 2, 4, 2), ActionWithTime(3, 2, 6, 6), ActionWithTime(2, 2, 5, 2), ActionWithTime(3, 2, 7, 6), ActionWithTime(2, 2, 6, 2), ActionWithTime(3, 2, 4, 2), ActionWithTime(2, 2, 7, 2), ActionWithTime(3, 2, 5, 2), ActionWithTime(3, 2, 6, 2), ActionWithTime(3, 2, 7, 2)], 
        [ActionWithTime(1, 3, 0, 3), ActionWithTime(1, 3, 1, 3), ActionWithTime(1, 3, 2, 3), ActionWithTime(1, 3, 3, 3), ActionWithTime(1, 3, 0, 7), ActionWithTime(2, 3, 0, 7), ActionWithTime(1, 3, 1, 7), ActionWithTime(2, 3, 1, 7), ActionWithTime(1, 3, 2, 7), ActionWithTime(2, 3, 2, 7), ActionWithTime(1, 3, 3, 7), ActionWithTime(2, 3, 3, 7), ActionWithTime(3, 3, 0, 7), ActionWithTime(1, 3, 4, 3), ActionWithTime(2, 3, 0, 3), ActionWithTime(3, 3, 1, 7), ActionWithTime(1, 3, 5, 3), ActionWithTime(2, 3, 1, 3), ActionWithTime(3, 3, 2, 7), ActionWithTime(1, 3, 6, 3), ActionWithTime(2, 3, 2, 3), ActionWithTime(3, 3, 3, 7), ActionWithTime(1, 3, 7, 3), ActionWithTime(2, 3, 3, 3), ActionWithTime(3, 3, 0, 3), ActionWithTime(1, 3, 4, 7), ActionWithTime(2, 3, 4, 7), ActionWithTime(3, 3, 1, 3), ActionWithTime(1, 3, 5, 7), ActionWithTime(2, 3, 5, 7), ActionWithTime(3, 3, 2, 3), ActionWithTime(1, 3, 6, 7), ActionWithTime(2, 3, 6, 7), ActionWithTime(3, 3, 3, 3), ActionWithTime(1, 3, 7, 7), ActionWithTime(2, 3, 7, 7), ActionWithTime(3, 3, 4, 7), ActionWithTime(2, 3, 4, 3), ActionWithTime(3, 3, 5, 7), ActionWithTime(2, 3, 5, 3), ActionWithTime(3, 3, 6, 7), ActionWithTime(2, 3, 6, 3), ActionWithTime(3, 3, 7, 7), ActionWithTime(2, 3, 7, 3), ActionWithTime(3, 3, 4, 3), ActionWithTime(3, 3, 5, 3), ActionWithTime(3, 3, 6, 3), ActionWithTime(3, 3, 7, 3)]]
    fwd_time = [7.12, 6.61, 6.65, 6.67, 7.12, 6.61, 6.65, 6.67]
    bwd_time = [15.1, 0, 0, 0, 15.1, 0, 0, 0]
    bwd_input_time = [10.7, 8.39, 10.41, 8.4, 10.7, 8.39, 10.41, 8.4]
    bwd_weight_time = [3.96, 4.79, 3.84, 4.94, 3.96, 4.79, 3.84, 4.94]
    pipeline_schedule = schedule_pipeline(link_actions(pipeline_schedule), fwd_time, bwd_time, bwd_input_time, bwd_weight_time)
    draw_pipeline_schedule("pipeline_schedule.pdf", pipeline_schedule)

    # draw_pipeline_schedule("pipeline_schedule.pdf", pipeline_schedule, fwd_time, bwd_time, bwd_input_time, bwd_weight_time)

    # 2-2 : uniform time version
    uniform_time = [10,10,10,10,10,10,10,10]
    pipeline_schedule = schedule_pipeline(link_actions(pipeline_schedule), uniform_time, [20,20,20,20,20,20,20,20], uniform_time, uniform_time)
    draw_pipeline_schedule("pipeline_schedule.pdf", pipeline_schedule)
    # draw_pipeline_schedule("pipeline_schedule.pdf", pipeline_schedule, uniform_time, [2,2,2,2], uniform_time, uniform_time)

    # Example pipeline schedule 3 : zbv, 4 ranks, 8 microbatches, 2 stages per rank
    pipeline_schedule = [[ActionWithTime(ActionType.FORWARD, 0, 0, 0), ActionWithTime(ActionType.FORWARD, 0, 1, 0), ActionWithTime(ActionType.FORWARD, 0, 2, 0), ActionWithTime(ActionType.FORWARD, 0, 3, 0), ActionWithTime(ActionType.FORWARD, 0, 4, 0), ActionWithTime(ActionType.FORWARD, 0, 5, 0), ActionWithTime(ActionType.FORWARD, 0, 6, 0), ActionWithTime(ActionType.FORWARD, 0, 0, 3), ActionWithTime(ActionType.BACKWARD_INPUT, 0, 0, 3), ActionWithTime(ActionType.BACKWARD_WEIGHT, 0, 0, 3), ActionWithTime(ActionType.FORWARD, 0, 1, 3), ActionWithTime(ActionType.BACKWARD_INPUT, 0, 1, 3), ActionWithTime(ActionType.BACKWARD_WEIGHT, 0, 1, 3), ActionWithTime(ActionType.FORWARD, 0, 2, 3), ActionWithTime(ActionType.BACKWARD_INPUT, 0, 2, 3), ActionWithTime(ActionType.BACKWARD_WEIGHT, 0, 2, 3), ActionWithTime(ActionType.FORWARD, 0, 3, 3), ActionWithTime(ActionType.BACKWARD_INPUT, 0, 3, 3), ActionWithTime(ActionType.BACKWARD_WEIGHT, 0, 3, 3), ActionWithTime(ActionType.FORWARD, 0, 7, 0), ActionWithTime(ActionType.FULL_BACKWARD, 0, 0, 0), ActionWithTime(ActionType.FORWARD, 0, 4, 3), ActionWithTime(ActionType.BACKWARD_INPUT, 0, 4, 3), ActionWithTime(ActionType.BACKWARD_WEIGHT, 0, 4, 3), ActionWithTime(ActionType.FULL_BACKWARD, 0, 1, 0), ActionWithTime(ActionType.FORWARD, 0, 5, 3), ActionWithTime(ActionType.BACKWARD_INPUT, 0, 5, 3), ActionWithTime(ActionType.BACKWARD_WEIGHT, 0, 5, 3), ActionWithTime(ActionType.FULL_BACKWARD, 0, 2, 0), ActionWithTime(ActionType.FORWARD, 0, 6, 3), ActionWithTime(ActionType.BACKWARD_INPUT, 0, 6, 3), ActionWithTime(ActionType.BACKWARD_WEIGHT, 0, 6, 3), ActionWithTime(ActionType.FULL_BACKWARD, 0, 3, 0), ActionWithTime(ActionType.FORWARD, 0, 7, 3), ActionWithTime(ActionType.BACKWARD_INPUT, 0, 7, 3), ActionWithTime(ActionType.BACKWARD_WEIGHT, 0, 7, 3), ActionWithTime(ActionType.FULL_BACKWARD, 0, 4, 0), ActionWithTime(ActionType.FULL_BACKWARD, 0, 5, 0), ActionWithTime(ActionType.FULL_BACKWARD, 0, 6, 0), ActionWithTime(ActionType.FULL_BACKWARD, 0, 7, 0)], 
                         [ActionWithTime(ActionType.FORWARD, 1, 0, 1), ActionWithTime(ActionType.FORWARD, 1, 1, 1), ActionWithTime(ActionType.FORWARD, 1, 2, 1), ActionWithTime(ActionType.FORWARD, 1, 3, 1), ActionWithTime(ActionType.FORWARD, 1, 4, 1), ActionWithTime(ActionType.FORWARD, 1, 0, 2), ActionWithTime(ActionType.FORWARD, 1, 5, 1), ActionWithTime(ActionType.FORWARD, 1, 1, 2), ActionWithTime(ActionType.BACKWARD_INPUT, 1, 0, 2), ActionWithTime(ActionType.BACKWARD_WEIGHT, 1, 0, 2), ActionWithTime(ActionType.FORWARD, 1, 2, 2), ActionWithTime(ActionType.BACKWARD_INPUT, 1, 1, 2), ActionWithTime(ActionType.BACKWARD_WEIGHT, 1, 1, 2), ActionWithTime(ActionType.FORWARD, 1, 3, 2), ActionWithTime(ActionType.BACKWARD_INPUT, 1, 2, 2), ActionWithTime(ActionType.BACKWARD_WEIGHT, 1, 2, 2), ActionWithTime(ActionType.FORWARD, 1, 6, 1), ActionWithTime(ActionType.BACKWARD_INPUT, 1, 0, 1), ActionWithTime(ActionType.BACKWARD_WEIGHT, 1, 0, 1), ActionWithTime(ActionType.FORWARD, 1, 4, 2), ActionWithTime(ActionType.BACKWARD_INPUT, 1, 3, 2), ActionWithTime(ActionType.BACKWARD_WEIGHT, 1, 3, 2), ActionWithTime(ActionType.FORWARD, 1, 7, 1), ActionWithTime(ActionType.BACKWARD_INPUT, 1, 1, 1), ActionWithTime(ActionType.BACKWARD_WEIGHT, 1, 1, 1), ActionWithTime(ActionType.FORWARD, 1, 5, 2), ActionWithTime(ActionType.BACKWARD_INPUT, 1, 4, 2), ActionWithTime(ActionType.BACKWARD_WEIGHT, 1, 4, 2), ActionWithTime(ActionType.BACKWARD_INPUT, 1, 2, 1), ActionWithTime(ActionType.BACKWARD_WEIGHT, 1, 2, 1), ActionWithTime(ActionType.FORWARD, 1, 6, 2), ActionWithTime(ActionType.BACKWARD_INPUT, 1, 5, 2), ActionWithTime(ActionType.BACKWARD_WEIGHT, 1, 5, 2), ActionWithTime(ActionType.BACKWARD_INPUT, 1, 3, 1), ActionWithTime(ActionType.BACKWARD_WEIGHT, 1, 3, 1), ActionWithTime(ActionType.FORWARD, 1, 7, 2), ActionWithTime(ActionType.BACKWARD_INPUT, 1, 6, 2), ActionWithTime(ActionType.BACKWARD_WEIGHT, 1, 6, 2), ActionWithTime(ActionType.BACKWARD_INPUT, 1, 4, 1), ActionWithTime(ActionType.BACKWARD_INPUT, 1, 7, 2), ActionWithTime(ActionType.BACKWARD_INPUT, 1, 5, 1), ActionWithTime(ActionType.BACKWARD_WEIGHT, 1, 4, 1), ActionWithTime(ActionType.BACKWARD_INPUT, 1, 6, 1), ActionWithTime(ActionType.BACKWARD_WEIGHT, 1, 5, 1), ActionWithTime(ActionType.BACKWARD_INPUT, 1, 7, 1), ActionWithTime(ActionType.BACKWARD_WEIGHT, 1, 6, 1), ActionWithTime(ActionType.BACKWARD_WEIGHT, 1, 7, 2), ActionWithTime(ActionType.BACKWARD_WEIGHT, 1, 7, 1)], 
                         [ActionWithTime(ActionType.FORWARD, 2, 0, 2), ActionWithTime(ActionType.FORWARD, 2, 1, 2), ActionWithTime(ActionType.FORWARD, 2, 2, 2), ActionWithTime(ActionType.FORWARD, 2, 0, 1), ActionWithTime(ActionType.FORWARD, 2, 3, 2), ActionWithTime(ActionType.FORWARD, 2, 1, 1), ActionWithTime(ActionType.FORWARD, 2, 4, 2), ActionWithTime(ActionType.FORWARD, 2, 2, 1), ActionWithTime(ActionType.BACKWARD_INPUT, 2, 0, 1), ActionWithTime(ActionType.BACKWARD_WEIGHT, 2, 0, 1), ActionWithTime(ActionType.FORWARD, 2, 3, 1), ActionWithTime(ActionType.BACKWARD_INPUT, 2, 1, 1), ActionWithTime(ActionType.BACKWARD_WEIGHT, 2, 1, 1), ActionWithTime(ActionType.FORWARD, 2, 5, 2), ActionWithTime(ActionType.BACKWARD_INPUT, 2, 0, 2), ActionWithTime(ActionType.BACKWARD_WEIGHT, 2, 0, 2), ActionWithTime(ActionType.FORWARD, 2, 4, 1), ActionWithTime(ActionType.BACKWARD_INPUT, 2, 2, 1), ActionWithTime(ActionType.BACKWARD_WEIGHT, 2, 2, 1), ActionWithTime(ActionType.FORWARD, 2, 6, 2), ActionWithTime(ActionType.BACKWARD_INPUT, 2, 1, 2), ActionWithTime(ActionType.BACKWARD_WEIGHT, 2, 1, 2), ActionWithTime(ActionType.FORWARD, 2, 5, 1), ActionWithTime(ActionType.BACKWARD_INPUT, 2, 3, 1), ActionWithTime(ActionType.BACKWARD_WEIGHT, 2, 3, 1), ActionWithTime(ActionType.FORWARD, 2, 7, 2), ActionWithTime(ActionType.BACKWARD_INPUT, 2, 2, 2), ActionWithTime(ActionType.BACKWARD_WEIGHT, 2, 2, 2), ActionWithTime(ActionType.FORWARD, 2, 6, 1), ActionWithTime(ActionType.BACKWARD_INPUT, 2, 4, 1), ActionWithTime(ActionType.BACKWARD_WEIGHT, 2, 4, 1), ActionWithTime(ActionType.BACKWARD_INPUT, 2, 3, 2), ActionWithTime(ActionType.BACKWARD_WEIGHT, 2, 3, 2), ActionWithTime(ActionType.FORWARD, 2, 7, 1), ActionWithTime(ActionType.BACKWARD_INPUT, 2, 5, 1), ActionWithTime(ActionType.BACKWARD_WEIGHT, 2, 5, 1), ActionWithTime(ActionType.BACKWARD_INPUT, 2, 4, 2), ActionWithTime(ActionType.BACKWARD_INPUT, 2, 6, 1), ActionWithTime(ActionType.BACKWARD_INPUT, 2, 5, 2), ActionWithTime(ActionType.BACKWARD_INPUT, 2, 7, 1), ActionWithTime(ActionType.BACKWARD_INPUT, 2, 6, 2), ActionWithTime(ActionType.BACKWARD_WEIGHT, 2, 4, 2), ActionWithTime(ActionType.BACKWARD_INPUT, 2, 7, 2), ActionWithTime(ActionType.BACKWARD_WEIGHT, 2, 5, 2), ActionWithTime(ActionType.BACKWARD_WEIGHT, 2, 6, 1), ActionWithTime(ActionType.BACKWARD_WEIGHT, 2, 7, 1), ActionWithTime(ActionType.BACKWARD_WEIGHT, 2, 6, 2), ActionWithTime(ActionType.BACKWARD_WEIGHT, 2, 7, 2)], 
                         [ActionWithTime(ActionType.FORWARD, 3, 0, 3), ActionWithTime(ActionType.FORWARD, 3, 0, 0), ActionWithTime(ActionType.FORWARD, 3, 1, 3), ActionWithTime(ActionType.FORWARD, 3, 1, 0), ActionWithTime(ActionType.FORWARD, 3, 2, 3), ActionWithTime(ActionType.FORWARD, 3, 2, 0), ActionWithTime(ActionType.FORWARD, 3, 3, 3), ActionWithTime(ActionType.FORWARD, 3, 3, 0), ActionWithTime(ActionType.FULL_BACKWARD, 3, 0, 0), ActionWithTime(ActionType.FORWARD, 3, 4, 3), ActionWithTime(ActionType.BACKWARD_INPUT, 3, 0, 3), ActionWithTime(ActionType.BACKWARD_WEIGHT, 3, 0, 3), ActionWithTime(ActionType.FORWARD, 3, 4, 0), ActionWithTime(ActionType.FULL_BACKWARD, 3, 1, 0), ActionWithTime(ActionType.FORWARD, 3, 5, 3), ActionWithTime(ActionType.BACKWARD_INPUT, 3, 1, 3), ActionWithTime(ActionType.BACKWARD_WEIGHT, 3, 1, 3), ActionWithTime(ActionType.FORWARD, 3, 5, 0), ActionWithTime(ActionType.FULL_BACKWARD, 3, 2, 0), ActionWithTime(ActionType.FORWARD, 3, 6, 3), ActionWithTime(ActionType.BACKWARD_INPUT, 3, 2, 3), ActionWithTime(ActionType.BACKWARD_WEIGHT, 3, 2, 3), ActionWithTime(ActionType.FORWARD, 3, 6, 0), ActionWithTime(ActionType.FULL_BACKWARD, 3, 3, 0), ActionWithTime(ActionType.FORWARD, 3, 7, 3), ActionWithTime(ActionType.BACKWARD_INPUT, 3, 3, 3), ActionWithTime(ActionType.BACKWARD_WEIGHT, 3, 3, 3), ActionWithTime(ActionType.FORWARD, 3, 7, 0), ActionWithTime(ActionType.FULL_BACKWARD, 3, 4, 0), ActionWithTime(ActionType.BACKWARD_INPUT, 3, 4, 3), ActionWithTime(ActionType.BACKWARD_INPUT, 3, 5, 3), ActionWithTime(ActionType.BACKWARD_INPUT, 3, 6, 3), ActionWithTime(ActionType.BACKWARD_INPUT, 3, 7, 3), ActionWithTime(ActionType.BACKWARD_WEIGHT, 3, 4, 3), ActionWithTime(ActionType.FULL_BACKWARD, 3, 5, 0), ActionWithTime(ActionType.FULL_BACKWARD, 3, 6, 0), ActionWithTime(ActionType.FULL_BACKWARD, 3, 7, 0), ActionWithTime(ActionType.BACKWARD_WEIGHT, 3, 5, 3), ActionWithTime(ActionType.BACKWARD_WEIGHT, 3, 6, 3), ActionWithTime(ActionType.BACKWARD_WEIGHT, 3, 7, 3)]]
    fwd_time, bwd_time, bwd_input_time, bwd_weight_time \
        = [8.3101, 10.3417,  9.5100,  7.3098], [14.8432,  0.0000,  0.0000, 13.5128], [11.4721, 16.9559, 10.7292,  9.1316], [10.2950, 26.8687, 15.5081,  7.1363]
    pipeline_schedule = schedule_pipeline(link_actions(pipeline_schedule), fwd_time, bwd_time, bwd_input_time, bwd_weight_time)
    draw_pipeline_schedule("pipeline_schedule.pdf", pipeline_schedule)
    # draw_pipeline_schedule("pipeline_schedule.pdf", pipeline_schedule, fwd_time, bwd_time, bwd_input_time, bwd_weight_time)