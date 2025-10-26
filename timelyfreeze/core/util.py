import os
from typing import List
from matplotlib import pyplot as plt
import numpy as np
from torchtitan.tools.logging import logger

from .action import ActionType, ActionWithTime
from .config import TimelyFreezeConfig

def get_abs_path(path:str, base_dir:str)->str:
    '''Get the absolute path of the specified path. 
    Args:
        path (str): the path to be joined with the default path.
        base_dir (str): base directory name to join if path is relative. (E.g. "/home/user/dump_folder" -> "/home/user/dump_folder/{path}")
    Returns:
        str: the absolute path.
    '''
    if not os.path.isabs(path):
        path = os.path.abspath(os.path.join(base_dir, path))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return path


def draw_line_chart(data_x, data_y, save_file, config: TimelyFreezeConfig, title=None, xlabel=None, ylabel=None, window_size=5):
    '''
    Draw the line chart of the data.
    Args:
        data_x: list of x-axis data.
        data_y: list of y-axis data.
        save_file: path to save the image.
        config: TimelyFreezeConfig object.
        title: title of the chart.
        xlabel: label of the x-axis.
        ylabel: label of the y-axis.
        window_size: window size for moving average trend line.
    '''
    if len(data_x) == 0 or len(data_y) == 0:
        logger.warning("Data is empty. Skip drawing the line chart.")
        return None
    elif len(data_x) != len(data_y):
        logger.warning(f"The length of data_x [{len(data_x)}] and data_y [{len(data_y)}] should be the same.")
        return None
    
    fig, ax = plt.subplots()
    # Fit and plot trend line
    if len(data_y) > 30:
        moving_avg = np.convolve(data_y, np.ones(window_size)/window_size, mode='valid')
        ax.plot(data_x[window_size-1:], moving_avg, linestyle='-', color="#DCD2C0", linewidth=8, alpha=0.5, zorder=1)
    ax.scatter(data_x, data_y, marker='o', color='#988456', s=6 if len(data_y) > 50 else 12, zorder=2)

    ax.set_title(title if title is not None else f"Line Chart of Rank {config.comm.global_rank} (Stage {config.comm.pp_rank})", 
                    fontdict={'fontsize': 13})
    if xlabel:
        ax.set_xlabel(xlabel, fontdict={'fontsize': 13})
    if ylabel:
        ax.set_ylabel(ylabel, fontdict={'fontsize': 13})
    save_file = get_abs_path(save_file, base_dir=config.metrics.image_folder)
    plt.savefig(save_file)
    plt.close()
    logger.info(f"{title if title is not None else 'Line Chart'}  is saved as: {save_file}")
    return save_file


def draw_elementwise_histogram(data, stage, save_file, config: TimelyFreezeConfig, title=None, xlabel1=None, xlabel2=None):
    '''
    Draw the elementwise histogram of the model.
    data: list of the count and total per element. [[count: int, total: int] for each element].    
    '''
    total_sum = sum([data_l[1] for data_l in data])
    if total_sum == 0:
        logger.warning("data is empty. Skip drawing the elementwise histogram.")
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
    bar_colors = np.linspace(bar_colors[0], bar_colors[1], len(data) * (config.parallelism.num_stages + 1))
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
    ax1.text(max(min(80, past_ratio-5),5), 0, total_count_text, ha='center', va='center', fontsize=11, color='black')
    total_elem_text = f'Total Sum\n{int(total_sum)}\n(100%)'
    ax1.text(95, 0, total_elem_text, ha='center', va='center', fontsize=11, color='black')
    ax1.set_xlim(0, 100)
    ax1.set_xticks(np.arange(0, 101, 10))
    ax1.set_yticks([])
    ax1.set_xlabel(f'{xlabel1 if xlabel1 else "Total Counts"} (%)')
    plt.subplots_adjust(hspace=0.6)

    ax2.set_xlim(-0.5, len(data) - 0.5)
    ax2.set_ylim(0, 1)
    xticks = range(len(data)) if len(data) < 20 else range(0, len(data), len(data)//20)
    ax2.set_xticks(xticks)
    ax2.set_xticklabels([f'{i}' for i in xticks], rotation=45, fontsize=9)
    ax2.set_ylabel('Count/Total Ratio')
    ax2.set_xlabel(f'{xlabel2 if xlabel2 else "Element Index"} (#)')

    fig.suptitle(title if title is not None else f"Elementwise Histogram of Rank {config.comm.global_rank} (Stage {stage})", fontsize=15)
    plt.subplots_adjust(bottom=0.2)

    save_file = get_abs_path(save_file, base_dir=config.metrics.image_folder)
    plt.savefig(save_file)
    plt.close()
    logger.info(f"Elementwise Histogram is saved as: {save_file}\n\t> Counts Sum: {int(past_counts)}, Total Sum: {int(total_sum)} ({past_counts/total_sum*100:.2f}%), " \
             + f"Ratio(%) per element: {[int(data_l[0] / data_l[1]*10000)/100 for data_l in data]}")
    return


def draw_pipeline_schedule(save_file:str, 
                            pipeline_schedule:List[List[ActionWithTime]],
                            config: TimelyFreezeConfig,
                            title=None, xlabel=None, ylabel=None):
    num_ranks = config.comm.pp
    stages_per_rank = [list(dict.fromkeys([action.stage for action in pipeline_schedule[rank] if action.type == ActionType.FORWARD and action.stage is not None])) for rank in range(num_ranks)]
    num_stages_per_rank = config.parallelism.stages_per_rank

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
    if max_time >= 2000:
        tick_unit = 500
    elif max_time >= 1000:
        tick_unit = 200
    elif max_time >= 400:
        tick_unit = 100
    else:
        tick_unit = 50
    # set the figure size and axes
    fig, ax = plt.subplots(figsize=(max(1, round(max_time/tick_unit*3)), 3), dpi=100)
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
    ax.set_xticks(np.append(np.arange(0, max_time if (max_time%tick_unit >= tick_unit*30) else max_time-tick_unit, tick_unit), max_time))
    ax.set_yticks(range(num_ranks))
    ax.set_yticklabels([f'Rank {i}' for i in range(num_ranks)], fontsize=20)
    ax.tick_params(axis='x', labelsize=20)
    ax.tick_params(axis='y', labelsize=20)

    if xlabel:
        ax.set_xlabel(xlabel, fontsize=20) # "Time"
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=20) # "Rank"
    if title is not None:
        ax.set_title(title, fontsize=18) # "Pipeline Schedule TimeBlock Sequence" if title is None else title

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
            if config.parallelism.bwd_separated and action.type == ActionType.FULL_BACKWARD:
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
                        ha='center', va='center', fontsize=18,
                        color=stage_color_map[action.rank].get(action.stage, 'black')
                    )
                    ax.text(
                        x=action._start_time + (action.duration + w_i)/2, y=action.rank, s=str(action.microbatch+1), # +1 for 1-based index
                        ha='center', va='center', fontsize=18,
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
                        ha='center', va='center', fontsize=18,
                        color=stage_color_map[action.rank].get(action.stage, 'black')
                    )
    
    save_file = get_abs_path(save_file, base_dir=config.metrics.image_folder)
    plt.savefig(save_file, bbox_inches='tight', pad_inches=0)
    plt.close()

    logger.info(f"Pipeline schedule is saved as: {save_file}\n> Batch Time: {up_time:.2f} ms, GPU Bubble Ratio: {', '.join([f'{val*100:.2f}%' for val in gpu_bubble_ratio])}")

    return
