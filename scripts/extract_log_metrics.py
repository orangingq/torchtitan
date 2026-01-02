#!/usr/bin/env python3
# filepath: /home/shcho/torchtitan/scripts/extract_log_metrics.py
"""Extract GPU Bubble Ratio quartets and Total Sum percentages from Torchtitan logs."""

from __future__ import annotations

import argparse
import glob
import os
import re
from datetime import datetime
from pathlib import Path
from statistics import fmean
from typing import Any, Dict, Iterable, List

import pandas as pd

# Regular expression patterns
# GPU Bubble Ratio per GPU -> 
# '''
# ...###_final....svg
# [rank#]: ... GPU Bubble Ratio: X1%, X2%, X3%, X4%"
GPU_LINE_PATTERN = re.compile(
    r"(?P<file>\d+_final[^\r\n]*\.svg)\r?\n\[(?P<rank>[^\]]+)\]:.*?GPU\s+Bubble\s+Ratio:\s*(?P<body>[^\r\n]+)",
    re.IGNORECASE | re.MULTILINE,
)
# Batch Time pattern -> "Batch Time: XX.XX ms"
# BATCH_TIME_PATTERN = re.compile(r"Batch\s+Time:\s*([0-9]+(?:\.[0-9]+)?)\s*ms", re.IGNORECASE)
# Percentage pattern -> "XX.XX%"
PERCENT_PATTERN = re.compile(r"([0-9]+(?:\.[0-9]+)?)%")
# Total Average Freeze Ratio -> 
# '''
# ..._stage<stage_num>.svg
# [rank#]: ... Counts Sum: NNN, Total Sum: MMM (<freeze_ratio>%)
# '''
TOTAL_SUM_PATTERN = re.compile(
    r"^\[(?P<rank>[^\]]+)\]:.*Counts\s+Sum:\s*(?P<counts>\d+),\s*Total\s+Sum:\s*(?P<total>\d+)\s*\((?P<freeze_ratio>[0-9]+(?:\.[0-9]+)?)%\)",
    re.IGNORECASE,
)
STAGE_FILENAME_PATTERN = re.compile(r"_stage(?P<stage>\d+)\.svg", re.IGNORECASE)


SUMMARY_COLUMNS = [
    "extracted_at",
    "file_path",
    "rank1_bubble_ratio",
    "rank2_bubble_ratio",
    "rank3_bubble_ratio",
    "rank4_bubble_ratio",
    "stage0_freeze_ratio",
    "stage1_freeze_ratio",
    "stage2_freeze_ratio",
    "stage3_freeze_ratio",
    "stage4_freeze_ratio",
    "stage5_freeze_ratio",
    "stage6_freeze_ratio",
    "stage7_freeze_ratio",
]


def parse_log_file(log_path: Path) -> Dict[str, List[Dict[str, float]]]:
    """각 로그 파일에서 GPU Bubble Ratio 1개와 stage별 Freeze Ratio를 수집한다."""
    gpu_rows: List[Dict[str, float]] = []
    total_rows: List[Dict[str, float]] = []
    current_stage: int | None = None

    content = log_path.read_text(encoding="utf-8")

    gpu_match = GPU_LINE_PATTERN.search(content)
    if gpu_match:
        ratios = [float(val) * 0.01 for val in PERCENT_PATTERN.findall(gpu_match.group("body"))]
        if len(ratios) == 4:
            gpu_rows.append(
                {
                    "rank": gpu_match.group("rank"),
                    "ratio_1": ratios[0],
                    "ratio_2": ratios[1],
                    "ratio_3": ratios[2],
                    "ratio_4": ratios[3],
                }
            )

    for line_no, line in enumerate(content.splitlines(), start=1):
        stage_marker = STAGE_FILENAME_PATTERN.search(line)
        if stage_marker:
            current_stage = int(stage_marker.group("stage"))

        total_match = TOTAL_SUM_PATTERN.search(line)
        if total_match:
            total_rows.append(
                {
                    "line": line_no,
                    "rank": total_match.group("rank"),
                    "counts_sum": int(total_match.group("counts")),
                    "total_sum": int(total_match.group("total")),
                    "freeze_ratio": float(total_match.group("freeze_ratio")) * 0.01,
                    "stage": current_stage,
                }
            )

    return {"gpu_bubble": gpu_rows, "total_sum": total_rows}


def last_value(values: Iterable[float | None]) -> float | None:
    cleaned = [value for value in values if value is not None]
    return round(float(cleaned[-1]), 4) if cleaned else None


def build_summary_row(log_path: Path, rows: Dict[str, List[Dict[str, float]]]) -> Dict[str, Any]:
    summary = {column: None for column in SUMMARY_COLUMNS}
    gpu_rows = rows.get("gpu_bubble", [])
    total_rows = rows.get("total_sum", [])

    summary["extracted_at"] = datetime.now().isoformat(timespec="seconds")
    summary["file_path"] = str(log_path)
    for idx in range(1, 5):
        summary[f"rank{idx}_bubble_ratio"] = last_value(entry.get(f"ratio_{idx}") for entry in gpu_rows)
    for stage_idx in range(8):
        stage_values = [
            entry.get("freeze_ratio")
            for entry in total_rows
            if entry.get("stage") == stage_idx
        ]
        summary[f"stage{stage_idx}_freeze_ratio"] = last_value(stage_values)

    return summary


def append_summary_rows(summary_rows: List[Dict[str, Any]], output_path: Path) -> None:
    if not summary_rows:
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(summary_rows, columns=SUMMARY_COLUMNS)
    write_header = not output_path.exists()

    if write_header:
        with output_path.open("w", encoding="utf-8") as handle:
            handle.write("\n")  # 첫 번째 행을 비워둠
        df.to_csv(output_path, mode="a", index=False, header=True)
    else:
        df.to_csv(output_path, mode="a", index=False, header=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract GPU Bubble Ratio quartets and Total Sum percentages from Torchtitan logs and summarize each file.",
    )
    parser.add_argument(
        "log_paths",
        type=str,
        nargs="*",
        help="Path(s) or glob pattern(s) to the log file(s) to parse.",
    )
    parser.add_argument(
        "--log_paths",
        dest="log_paths_option",
        type=str,
        nargs="+",
        help="Additional path(s) or glob pattern(s) to include.",
    )
    parser.add_argument("--log_dir", type=Path, help="Directory Path to the log files to parse.")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Path to the output CSV file (default: <common log directory>/log_metrics_extracted_results.csv).",
    )
    return parser.parse_args()


def resolve_log_files(raw_patterns: List[str]) -> List[Path]:
    """Resolve and return a list of log files based on the provided glob-like patterns."""
    log_files: List[Path] = []
    seen: set[Path] = set()

    for pattern in raw_patterns:
        expanded_pattern = str(Path(pattern).expanduser())
        matches = glob.glob(expanded_pattern, recursive=True)
        if not matches:
            candidate = Path(expanded_pattern).expanduser().resolve()
            if candidate.is_file() and candidate not in seen:
                seen.add(candidate)
                log_files.append(candidate)
            continue
        for match in matches:
            candidate = Path(match).expanduser().resolve()
            if candidate.is_file() and candidate not in seen:
                seen.add(candidate)
                log_files.append(candidate)

    log_files.sort()
    return log_files


def determine_output_path(log_files: List[Path], user_output: Path | None) -> Path:
    if user_output:
        return user_output.expanduser().resolve()

    common_dir = Path(os.path.commonpath([str(path.parent) for path in log_files]))
    return common_dir / "log_metrics_extracted_results.csv"


def main() -> None:
    args = parse_args()
    raw_patterns: List[str] = []
    if args.log_paths:
        raw_patterns.extend(args.log_paths)
    if args.log_paths_option:
        raw_patterns.extend(args.log_paths_option)

    log_files = resolve_log_files(raw_patterns)

    if not log_files and args.log_dir:
        log_dir_path = args.log_dir.expanduser().resolve()
        if not log_dir_path.is_dir():
            raise NotADirectoryError(f"Log directory not found: {log_dir_path}")
        log_files = sorted(path for path in log_dir_path.iterdir() if path.is_file())

    if not log_files:
        raise FileNotFoundError("No log files matched the provided inputs.")

    output_path = determine_output_path(log_files, args.output)

    summary_rows: List[Dict[str, Any]] = []
    for log_path in log_files:
        rows = parse_log_file(log_path)
        summary_row = build_summary_row(log_path, rows)
        summary_rows.append(summary_row)
        print(
            f"[{log_path.name}] Found {len(rows['gpu_bubble'])} GPU Bubble Ratio quartet rows and {len(rows['total_sum'])} Total Sum rows.",
        )

    append_summary_rows(summary_rows, output_path)
    print(f"Appended {len(summary_rows)} row(s) to {output_path}.")


if __name__ == "__main__":
    main()