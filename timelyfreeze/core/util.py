import os
from typing import List, Tuple
import matplotlib
from matplotlib import pyplot as plt
import numpy as np
from torchtitan.tools.logging import logger
from typing import List

# Set default matplotlib parameters for consistent styling
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['DejaVu Serif', 'Helvetica', 'Arial', 'DejaVu Sans']
matplotlib.rcParams['svg.fonttype'] = 'path'
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype']  = 42
matplotlib.rcParams['mathtext.fontset'] = 'cm' # TeX Computer Modern 느낌

default_color_palette: dict[str, str] = {
    "black": "#000000",
    "dark_gray": "#757575",
    "light_gray": "#C9C9C9",
    "dark_blue": "#0B3D91",
    "blue": "#166CF7",
    "middle_blue": "#64a5fa",
    "light_blue": "#b2d3ff",
    "very_light_blue": "#e6f0ff",
    "green": "#20A443",
    "jade_green": "#2FB8C2",
    "dark_gold": "#433b26",
    "gold": "#988456",
    "light_gold": "#DCD2C0",
    "off_white": "#f6f4ef",
    "red": "#EA4335",
}

from .action import ActionType, ActionWithTime, ActionWithFreezing
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
    else:
        path = os.path.normpath(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return path


def draw_line_chart(
    data_x,
    data_y,
    save_file,
    config: TimelyFreezeConfig,
    title=None,
    xlabel=None,
    ylabel=None,
    window_size=5,
    figsize: tuple[float, float] = (10, 6),
    line_color: str = default_color_palette["light_blue"],
    scatter_color: str = default_color_palette["blue"],
    scatter_size: int = 12,
    line_width: float = 5.0,
    alpha: float = 0.7,
    title_fontsize: int = 20,
    label_fontsize: int = 18,
    tick_fontsize: int = 18,
):
    """
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
        figsize: size of the figure (width, height).
        line_color: color of the trend line.
        scatter_color: color of the scatter points.
        scatter_size: size of the scatter points.
        line_width: width of the trend line.
        alpha: transparency level of the trend line.
        title_fontsize: font size of the title.
        label_fontsize: font size of the axis labels.
        tick_fontsize: font size of the tick labels.
    """
    if len(data_x) == 0 or len(data_y) == 0:
        logger.warning("Data is empty. Skip drawing the line chart.")
        return None
    elif len(data_x) != len(data_y):
        logger.warning(f"The length of data_x [{len(data_x)}] and data_y [{len(data_y)}] should be the same.")
        return None

    fig, ax = plt.subplots(figsize=figsize)
    # Fit and plot trend line
    n = len(data_y)
    if n > 30:
        w = max(1, min(window_size, n))
        kernel = np.ones(w, dtype=float) / w
        moving_avg = np.convolve(np.asarray(data_y, dtype=float), kernel, mode='valid')
        ax.plot(
            data_x[w - 1:],
            moving_avg,
            linestyle='-',
            color=line_color,
            linewidth=line_width,
            alpha=alpha,
            zorder=1,
        )
    ax.scatter(
        data_x,
        data_y,
        marker='o',
        color=scatter_color,
        s=scatter_size if len(data_y) > 50 else scatter_size * 2,
        zorder=2,
    )

    ax.set_title(
        title if title is not None else f"Line Chart of Rank {config.comm.global_rank} (Stage {config.comm.pp_rank})",
        fontsize=title_fontsize,
    )
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=label_fontsize)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=label_fontsize)
    ax.tick_params(axis='both', labelsize=tick_fontsize)

    save_file = get_abs_path(save_file, base_dir=config.metrics.image_folder)
    fig.savefig(save_file)
    plt.close(fig)
    logger.info(f"{title if title is not None else 'Line Chart'} is saved as: {save_file}")
    return save_file

def draw_afr_time_scatter_plot(
    afrs: List[float],
    times: List[float],
    save_file: str,
    config: TimelyFreezeConfig,
    microbatches: List[float] | None = None,
    title: str | None = None,
    xlabel: str = 'Freeze Ratio',
    ylabel: str = 'Time (ms)',
    figsize: Tuple[float, float] = (3.5, 3.5),
    line_color: str = default_color_palette["blue"],
    scatter_color: str = default_color_palette["middle_blue"],
    microbatch_base_color: str = default_color_palette["light_blue"],
    scatter_size: int = 10,
    linewidth: float = 1.1,
    grid_alpha: float = 0.5,
    label_fontsize: int = 10,
    tick_fontsize: int = 10,
    title_fontsize: int = 10,
    legend_fontsize: int = 10,
    xticks: List[float] = [0, 0.25, 0.5, 0.75, 1],
    yticks: List[float] | None = None,
) -> Tuple[float, float]:
    """
    Draw the trend line for the monitored values and save the plot.
    Returns slope/intercept of linear fit for later use.
    Args:
        afrs (List[float]): List of AFR (Active Freezing Ratio) values.
        times (List[float]): List of corresponding time measurements.
        save_file (str): Destination path (relative to config.metrics.image_folder).
        config (TimelyFreezeConfig): Run configuration.
        microbatches (List[float] | None): Optional list of microbatch indices for color-coding.
        title (str | None): Figure title.
        figsize (Tuple[float, float]): Figure size.
        line_color (str): Color of the trend line.
        scatter_color (str): Color of the scatter points.
        scatter_size (int): Size of the scatter points.
        linewidth (float): Width of the trend line.
        grid_alpha (float): Transparency level of the grid lines.
        label_fontsize (int): Font size for axis labels.
        tick_fontsize (int): Font size for tick labels.
        title_fontsize (int): Font size for the figure title.
        legend_fontsize (int): Font size for the legend.
        xticks (List[float]): X-axis tick positions.
        yticks (List[float] | None): Y-axis tick positions.
        microbatch_base_color (str): Hex color (e.g., '#RRGGBB') used as the base for microbatch gradients.
    """
    if len(afrs) == 0 or len(times) == 0:
        logger.warning("AFR/time data is empty. Skip drawing the scatter plot.")
        return 0.0, 0.0
    elif len(afrs) != len(times):
        logger.warning(f"The length of afrs [{len(afrs)}] and times [{len(times)}] should be the same.")
        return 0.0, 0.0
    
    fig, ax = plt.subplots(figsize=figsize)

    # Linear regression between AFRs and times.
    afr_buckets = {}
    for afr, time in zip(afrs, times):
        afr_buckets.setdefault(afr, []).append(time)

    unique_afrs = np.array(sorted(afr_buckets.keys()), dtype=float)
    averaged_times = np.array([float(np.mean(afr_buckets[afr])) for afr in unique_afrs], dtype=float)

    if len(unique_afrs) == 1:
        a, b = 0.0, averaged_times[0]
    else:
        a, b = np.polyfit(unique_afrs, averaged_times, 1)
    trend_fn = lambda r: a * r + b

    # Configure axis ranges.
    ax.set_xlim(0, 1)
    y_min, y_max = float(np.min(times)), float(np.max(times))
    ax.set_ylim(y_min, y_max)

    # Plot regression line.
    r_range = np.linspace(0, 1, 100)
    t_range = trend_fn(r_range)
    ax.plot(r_range, t_range, linestyle='-', color=line_color, linewidth=linewidth, label=f't = {a:.2f}r + {b:.2f}')

    # Scatter AFR/time pairs, optionally color-coded by microbatch.
    if microbatches is not None and len(microbatches) == len(times):
        mb_arr = np.array(microbatches, dtype=float)
        mb_min, mb_max = float(mb_arr.min()), float(mb_arr.max())
        norm = (mb_arr - mb_min) / (mb_max - mb_min) if mb_max > mb_min else np.zeros_like(mb_arr)
        base = np.array(matplotlib.colors.to_rgb(microbatch_base_color), dtype=float)
        factors = 1.0 - 0.6 * norm
        colors = np.clip(base[None, :] * factors[:, None], 0.0, 1.0)
        ax.scatter(afrs, times, c=colors, marker='o', s=scatter_size)
    else:
        ax.scatter(afrs, times, color=scatter_color, marker='o', s=scatter_size)

    # Cosmetic settings.
    ax.grid(True, linestyle='--', alpha=grid_alpha)
    ax.set_xticks(xticks)
    ax.set_yticks(yticks or [y_min, y_min * 0.75 + y_max * 0.25, (y_min + y_max) / 2, y_min * 0.25 + y_max * 0.75, y_max])
    ax.tick_params(axis='both', labelsize=tick_fontsize)
    ax.set_xlabel(xlabel, fontsize=label_fontsize)
    ax.set_ylabel(ylabel, fontsize=label_fontsize)
    ax.legend(loc='upper left', fontsize=legend_fontsize)
    if title:
        ax.set_title(title, fontsize=title_fontsize)

    # Remove chart borders for a cleaner look.
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Persist figure.
    save_file = get_abs_path(save_file, base_dir=config.metrics.image_folder)
    fig.savefig(save_file)
    plt.close(fig)
    logger.info(f"{title or 'Scatter Plot'} is saved as: {save_file}")
    return a, b


def draw_elementwise_histogram(
    data,
    stage,
    save_file,
    config: TimelyFreezeConfig,
    title=None,
    xlabel1=None,
    xlabel2=None,
    bar_width: float = 0.8,
    max_xticks: int = 20,
    figsize: tuple[float, float] | None = None,
    fontsize: int = 14,
    ticklabel_fontsize: int = 14,
    title_fontsize: int = 16,
    color_start: str = default_color_palette["very_light_blue"],
    color_end: str = default_color_palette["dark_blue"],
):
    """
    Draw stacked total-count and per-element histograms.

    Args:
        data (List[Tuple[int, int]]): Each entry is (count, total) for one element.
        stage (int): Stage index whose slice should be colorized.
        save_file (str): Destination path (relative to config.metrics.image_folder).
        config (TimelyFreezeConfig): Run configuration.
        title (str | None): Figure title.
        xlabel1 (str | None): Label for the cumulative ratio axis.
        xlabel2 (str | None): Label for the per-element axis.
        bar_width (float): Thickness of the stacked bar in the top subplot.
        max_xticks (int): Maximum number of tick labels displayed on the bottom axis.
        figsize (tuple[float, float] | None): Optional manual figure size override.
        fontsize (int): Font size for text annotations inside bars.
        ticklabel_fontsize (int): Font size for x-axis tick labels.
        title_fontsize (int): Font size for the figure title.
        color_start (str): Hex color for the gradient start.
        color_end (str): Hex color for the gradient end.
    """
    total_sum = sum(data_l[1] for data_l in data)
    if total_sum == 0:
        logger.warning("data is empty. Skip drawing the elementwise histogram.")
        return None

    num_elements = len(data)
    fig_size = figsize or ((10, 4) if num_elements < max_xticks else (15, 4))
    fig, (ax1, ax2) = plt.subplots(
        2,
        1,
        figsize=fig_size,
        gridspec_kw={"height_ratios": [1, 3]},
    )

    # Smooth gradient colors per stage to highlight distribution differences.
    color_start = np.array(matplotlib.colors.to_rgb(color_start)) # np.array([246, 244, 239]) / 255.0
    color_end   = np.array(matplotlib.colors.to_rgb(color_end)) # np.array([67, 59, 38]) / 255.0
    bar_colors = np.linspace(
        color_start, color_end, num_elements * (config.parallelism.num_stages + 1),
    )[stage * num_elements : (stage + 1) * num_elements]

    past_counts = 0
    for i, ((count, total), color) in enumerate(zip(data, bar_colors)):
        ratio = count / total_sum * 100
        past_ratio = past_counts / total_sum * 100

        # Top subplot: cumulative percentage of counts.
        ax1.barh(0, ratio, height=bar_width / 3, color=color, edgecolor="none", left=past_ratio)

        # Bottom subplot: count/total for each element.
        ratio_per_element = count / total
        ax2.bar(i, ratio_per_element, color=color, edgecolor="none")

        past_counts += count

    # 
    total_count_text = f"{past_counts/total_sum*100:.2f}%\n({int(past_counts)})"
    ax1.text(max(min(80, past_ratio - 5), 5), 0, total_count_text, ha="center", va="center", fontsize=fontsize, color="black")
    total_elem_text = f"100%\n({int(total_sum)})"
    ax1.text(95, 0, total_elem_text, ha="center", va="center", fontsize=fontsize, color="black")
    ax1.set_xlim(0, 100)
    ax1.set_xticks(np.arange(0, 101, 10))
    ax1.set_yticks([])
    ax1.set_xlabel(f"{xlabel1 if xlabel1 else 'Total Counts'} (%)")

    ax2.set_xlim(-0.5, num_elements - 0.5)
    ax2.set_ylim(0, 1)
    xtick_step = int(num_elements / max_xticks) if num_elements > max_xticks else 1
    xticks = range(0, num_elements, xtick_step)
    ax2.set_xticks(xticks)
    ax2.set_xticklabels([f"{i}" for i in xticks], rotation=45, fontsize=ticklabel_fontsize)
    ax2.set_ylabel("Count/Total Ratio")
    ax2.set_xlabel(f"{xlabel2 if xlabel2 else 'Element Index'} (#)")

    fig.suptitle(
        title if title is not None else f"Elementwise Histogram of Rank {config.comm.global_rank} (Stage {stage})",
        fontsize=title_fontsize,
    )
    plt.subplots_adjust(hspace=0.6, bottom=0.2)

    save_file = get_abs_path(save_file, base_dir=config.metrics.image_folder)
    fig.savefig(save_file)
    plt.close(fig)
    logger.info(
        "Elementwise Histogram is saved as: %s\n\t> Counts Sum: %d, Total Sum: %d (%.2f%%), Ratio(%%) per element: %s",
        save_file,
        int(past_counts),
        int(total_sum),
        past_counts / total_sum * 100,
        [int(data_l[0] / data_l[1] * 10000) / 100 for data_l in data],
    )
    return

def draw_pipeline_schedule(
        save_file: str,
        pipeline_schedule: List[List[ActionWithTime]],
        config: TimelyFreezeConfig,
        title: str | None = None,
        xlabel: str | None = None,
        ylabel: str | None = None,
        tick_unit: int | None = None,
        color_map: dict[ActionType, str] = {
                ActionType.FORWARD: default_color_palette["blue"],
                ActionType.FULL_BACKWARD: default_color_palette["green"],
                ActionType.BACKWARD_INPUT: default_color_palette["jade_green"],
                ActionType.BACKWARD_WEIGHT: default_color_palette["green"],
            },
        label_fontsize: int = 20,
        tick_fontsize: int = 20,
        title_fontsize: int = 20,
        microbatch_fontsize: int = 20,
        grid_linewidth: float = 0.7,
        grid_alpha: float = 0.7,
    ):
        """
        Visualize the pipeline execution timeline for each rank based on the collected schedule
        information, and save the resulting plot plus summary statistics to disk.

        Args:
            save_file (str): Relative or absolute path (w.r.t. `config.metrics.image_folder`)
                where the rendered schedule image will be stored.
            pipeline_schedule (List[List[ActionWithTime]]): Nested collection of per-rank
                actions describing their start/end times, types, stages, and microbatches.
            config (TimelyFreezeConfig): Run configuration containing communication and
                parallelism details needed for layout, coloring, and metadata.
            title (str | None): Optional plot title placed above the schedule.
            xlabel (str | None): Optional label for the horizontal time axis.
            ylabel (str | None): Optional label for the vertical rank axis.
            tick_unit (int | None): Explicit spacing (in milliseconds) between x-axis ticks;
                inferred heuristically from the total time span when omitted.
            color_map (dict[ActionType, str] | None): Per-action color overrides; defaults to
                predefined forward/backward hues when not provided.
            label_fontsize (int): Font size for axis labels.
            tick_fontsize (int): Font size for tick labels on both axes.
            title_fontsize (int): Font size for the plot title.
            microbatch_fontsize (int): Font size for the microbatch identifiers drawn inside bars.
            grid_linewidth (float): Line width of the auxiliary vertical grid lines.
            grid_alpha (float): Transparency level of the grid lines.

        Returns:
            None
        """
        num_ranks = config.comm.pp
        # Collect stages per rank to colorize them consistently.
        stages_per_rank = [
            list(
                dict.fromkeys(
                    [
                        action.stage
                        for action in pipeline_schedule[rank]
                        if action.type == ActionType.FORWARD and action.stage is not None
                    ]
                )
            )
            for rank in range(num_ranks)
        ]
        num_stages_per_rank = config.parallelism.stages_per_rank

        max_time = max(actions_per_rank[-1].end_time for actions_per_rank in pipeline_schedule)
        space = 0
        stage_color_palette = [
            f"#{int(255 - s / (num_stages_per_rank - 0.999) * 255):02X}"
            f"{int(255 - s / (num_stages_per_rank - 0.999) * 255):02X}"
            f"{int(255 - s / (num_stages_per_rank - 0.999) * 255):02X}"
            for s in range(num_stages_per_rank or 1)
        ]
        # Map each (rank, stage) pair to a color for text labels.
        stage_color_map = []
        for rank in range(num_ranks):
            color_lookup = {}
            for idx, stage in enumerate(stages_per_rank[rank]):
                color_lookup[stage] = stage_color_palette[idx % len(stage_color_palette)]
            stage_color_map.append(color_lookup)

        # Choose tick spacing heuristically based on the total time span.
        if tick_unit is None:
            tick_candidates = [50, 100, 200, 500]
            tick_unit = min(tick_candidates, key=lambda v: (abs(v - int(max_time // 4)), v))

        fig, ax = plt.subplots(figsize=(max(1, round(max_time / tick_unit * 3)), 3), dpi=100)
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        plt.margins(0)

        if not (xlabel or ylabel):
            ax.axis("off")
        ax.spines["bottom"].set_position(("outward", 0))
        ax.spines["top"].set_color("none")
        ax.spines["right"].set_color("none")
        ax.xaxis.set_ticks_position("bottom")
        ax.yaxis.set_ticks_position("left")
        ax.set_xlim(-space, max_time + space)
        ax.invert_yaxis()
        ax.set_xticks(
            np.append(
                np.arange(0, max_time if (max_time % tick_unit >= tick_unit * 30) else max_time - tick_unit, tick_unit),
                max_time,
            )
        )
        ax.set_yticks(range(num_ranks))
        ax.set_yticklabels([f"Rank {i}" for i in range(num_ranks)], fontsize=tick_fontsize)
        ax.tick_params(axis="x", labelsize=tick_fontsize)
        ax.tick_params(axis="y", labelsize=tick_fontsize)

        if xlabel:
            ax.set_xlabel(xlabel, fontsize=label_fontsize)
        if ylabel:
            ax.set_ylabel(ylabel, fontsize=label_fontsize)
        if title is not None:
            ax.set_title(title, fontsize=title_fontsize)

        # Draw vertical helper grid lines for readability.
        for time in ax.get_xticks():
            ax.axvline(x=time, color="gray", linestyle="--", linewidth=grid_linewidth, alpha=grid_alpha, zorder=0)

        up_time = max(actions_per_rank[-1].end_time - actions_per_rank[0].start_time for actions_per_rank in pipeline_schedule)
        gpu_util_time = [sum(action.duration for action in actions_per_rank) for actions_per_rank in pipeline_schedule]
        gpu_bubble_ratio = [(1 - gpu_util_time[rank] / up_time) for rank in range(num_ranks)]

        str_info = ""
        recent_n = 10
        action_type_name = {
            ActionType.FORWARD: "F",
            ActionType.FULL_BACKWARD: "B",
            ActionType.BACKWARD_INPUT: "BI",
            ActionType.BACKWARD_WEIGHT: "BW",
        }
        for actions_per_rank in pipeline_schedule:
            action_info = []
            for action in actions_per_rank:
                entry = (
                    f"[S{action.stage},MB{action.microbatch},{action_type_name[action.type]}]".ljust(12)
                    + f"{int(action.start_time)}-{int(action.duration)}-{int(action.end_time)}ms".ljust(14)
                )

                if isinstance(action, ActionWithFreezing) and len(action.frozen_ratio_history) > recent_n:
                    mean_recent = float(np.mean(np.asarray(action.frozen_ratio_history[-recent_n:], dtype=float)))
                    entry += f" AFR({recent_n}):{mean_recent:.4f}".ljust(15)
                action_info.append(entry)

                # If backward is split, paint input/weight separately, otherwise draw a single block.
                if config.parallelism.bwd_separated and action.type == ActionType.FULL_BACKWARD:
                    w_i = action.min_duration if hasattr(action, "min_duration") else action.duration / 2
                    ax.barh(
                        y=action.rank,
                        width=w_i,
                        left=action.start_time,
                        height=1,
                        color=color_map[ActionType.BACKWARD_INPUT],
                        edgecolor="white",
                        label=action.microbatch,
                    )
                    ax.barh(
                        y=action.rank,
                        width=action.duration - w_i,
                        left=action.start_time + w_i,
                        height=1,
                        color=color_map[ActionType.BACKWARD_WEIGHT],
                        edgecolor="white",
                        label=action.microbatch,
                    )
                    if action.duration > 1:
                        ax.text(
                            x=action.start_time + w_i / 2,
                            y=action.rank,
                            s=str(action.microbatch + 1),
                            ha="center",
                            va="center",
                            fontsize=microbatch_fontsize,
                            color=stage_color_map[action.rank].get(action.stage, "black"),
                        )
                        ax.text(
                            x=action.start_time + (action.duration + w_i) / 2,
                            y=action.rank,
                            s=str(action.microbatch + 1),
                            ha="center",
                            va="center",
                            fontsize=microbatch_fontsize,
                            color=stage_color_map[action.rank].get(action.stage, "black"),
                        )
                else:
                    ax.barh(
                        y=action.rank,
                        width=action.duration,
                        left=action.start_time,
                        height=1,
                        color=color_map[action.type],
                        edgecolor="white",
                        label=action.microbatch,
                    )
                    if action.duration > 1:
                        ax.text(
                            x=action.start_time + action.duration / 2,
                            y=action.rank,
                            s=str(action.microbatch + 1),
                            ha="center",
                            va="center",
                            fontsize=microbatch_fontsize,
                            color=stage_color_map[action.rank].get(action.stage, "black"),
                        )
            str_info += f"\n\t\t\t  [Rank {actions_per_rank[0].rank}]\t" + " | ".join(action_info)

        save_file = get_abs_path(save_file, base_dir=config.metrics.image_folder)
        fig.savefig(save_file, bbox_inches="tight", pad_inches=0)
        plt.close(fig)

        str_info = (
            f"\nPipeline schedule is saved as: {save_file}"
            + f"\n\t\t\t> Batch Time: {up_time:.2f} ms, GPU Bubble Ratio: "
            + ", ".join([f"{val*100:.2f}%" for val in gpu_bubble_ratio])
            + str_info
        )
        logger.info(str_info)

        return
