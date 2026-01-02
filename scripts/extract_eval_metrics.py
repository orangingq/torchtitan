#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List

import pandas as pd

SUMMARY_COLUMNS = [
    "extracted_at",
    "file_path",
    "mmlu",
    "hellaswag",
    "arc_challenge",
    "truthfulqa_mc1",
]

METRIC_PATTERN = re.compile(
    r"^(?P<name>mmlu|hellaswag|arc_challenge|truthfulqa_mc1)\s*\|\s*(?P<value>[0-9]+(?:\.[0-9]+)?)%",
    re.IGNORECASE,
)


def parse_log_file(log_path: Path) -> Dict[str, float | None]:
    values = {column: None for column in SUMMARY_COLUMNS[2:]}
    for line in log_path.read_text(encoding="utf-8").splitlines():
        match = METRIC_PATTERN.match(line.strip())
        if not match:
            continue
        metric = match.group("name").lower()
        values[metric] = round(float(match.group("value"))*0.01, 4)  # 백분율을 소수로 변환
    return values


def build_summary_row(log_path: Path, metrics: Dict[str, float | None]) -> Dict[str, Any]:
    row = {column: None for column in SUMMARY_COLUMNS}
    row["extracted_at"] = datetime.now().isoformat(timespec="seconds")
    row["file_path"] = str(log_path)
    for name in SUMMARY_COLUMNS[2:]:
        row[name] = metrics.get(name)
    return row


def append_rows(rows: List[Dict[str, Any]], output_path: Path) -> None:
    if not rows:
        return
    df = pd.DataFrame(rows, columns=SUMMARY_COLUMNS)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not output_path.exists()
    df.to_csv(output_path, mode="a", index=False, header=write_header)


def resolve_log_files(patterns: Iterable[str]) -> List[Path]:
    files: List[Path] = []
    seen: set[Path] = set()
    for pattern in patterns:
        expanded = str(Path(pattern).expanduser())
        matches = glob.glob(expanded, recursive=True) or [expanded]
        for match in matches:
            candidate = Path(match).expanduser().resolve()
            if candidate.is_file() and candidate not in seen:
                seen.add(candidate)
                files.append(candidate)
    files.sort()
    return files


def determine_output_path(log_files: List[Path], override: Path | None) -> Path:
    if override:
        return override.expanduser().resolve()
    common_dir = Path(os.path.commonpath([str(path.parent) for path in log_files]))
    return common_dir / "eval_metrics_extracted_results.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract eval metrics (mmlu, hellaswag, arc_challenge, truthfulqa_mc1) from Torchtitan eval logs."
    )
    
    parser.add_argument(
        "log_paths",
        type=str,
        nargs="+",
        help="Path(s) or glob pattern(s) to eval log files (e.g., logs/.../eval/eval_1224_*.log).",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Optional CSV output path (default: <common log directory>/eval_metrics_extracted_results.csv).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    log_files = resolve_log_files(args.log_paths)
    if not log_files:
        raise FileNotFoundError("No log files matched the provided inputs.")
    output_path = determine_output_path(log_files, args.output)

    rows: List[Dict[str, Any]] = []
    for log_path in log_files:
        metrics = parse_log_file(log_path)
        if all(value is None for value in metrics.values()):
            print(f"[WARN] {log_path} 에서 대상 지표를 찾지 못했습니다.")
            continue
        rows.append(build_summary_row(log_path, metrics))
        print(
            f"[OK] {log_path.name}: "
            + ", ".join(f"{name}={metrics.get(name)}" for name in SUMMARY_COLUMNS[2:])
        )

    append_rows(rows, output_path)
    if rows:
        print(f"{len(rows)}개 로그 요약을 {output_path}에 추가했습니다.")


if __name__ == "__main__":
    main()
