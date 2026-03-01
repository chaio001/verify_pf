#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import multiprocessing
import re
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

# =============================================================================
# 1. 核心配置与数据结构
# =============================================================================

DEFAULT_DIR = r"/mnt/sda/cyhcpp/Pythia/a_pythia/ChampSim-master/results"
WARMUP_CYCLES = 10_000_000
INTERVAL_SIZE = 1_000_000

# 匹配文件名
FILENAME_RE = re.compile(
    r"^(.*?)-hashed_perceptron-.*-(pathfinder|from_file)-.*_(\d+)_(\d+)\.csv$"
)

@dataclass
class CsvRecord:
    benchmark: str
    model: str
    size: int
    freq: int
    
    total_changes: int
    valid_cycles: int
    change_ratio: float
    
    # 记录每个 1M 区间的切换次数
    interval_changes: dict 

# =============================================================================
# 2. CSV 解析核心逻辑
# =============================================================================

def process_csv(filepath: Path) -> Optional[CsvRecord]:
    filename = filepath.name
    m = FILENAME_RE.match(filename)
    if not m:
        return None
    
    benchmark = m.group(1)
    model_raw = m.group(2)
    size = int(m.group(3))
    freq = int(m.group(4))
    
    model_name = "pythia" if model_raw == "from_file" else "pathfinder"
    target_action_col = "RLAction" if model_name == "pythia" else "Action"

    history = {}  
    total_changes = 0
    valid_cycles = 0  # <--- 新逻辑：用于记录 10M 之后的总访存指令行数
    interval_counts = defaultdict(int)

    try:
        with filepath.open("r", encoding="utf-8") as f:
            reader = csv.reader(f)
            try:
                header = next(reader)
            except StopIteration:
                return None
            
            try:
                idx_cycle = header.index("Cycle")
                idx_d1 = header.index("Delta1")
                idx_d2 = header.index("Delta2")
                idx_d3 = header.index("Delta3")
                idx_issued = header.index("IsIssued")
                idx_exploited = header.index("IsExploited")
                idx_target = header.index(target_action_col)
            except ValueError:
                return None 

            for row in reader:
                if not row:
                    continue
                
                cycle = int(row[idx_cycle])
                
                # 1. 跳过前 10M 周期的 Warmup
                if cycle < WARMUP_CYCLES:
                    continue
                
                # 2. 【核心修改】只要 Cycle >= 10M，不论后续状态如何，都算作一条 Valid 访存指令
                valid_cycles += 1
                
                # 3. 只有 IsIssued=1 且 IsExploited=1 的指令，才参与 Action 切换的统计
                if row[idx_issued] != '1' or row[idx_exploited] != '1':
                    continue
                
                feature = (row[idx_d1], row[idx_d2], row[idx_d3])
                current_action = row[idx_target]
                
                if feature in history:
                    if history[feature] != current_action:
                        total_changes += 1
                        history[feature] = current_action
                        
                        interval_idx = (cycle - WARMUP_CYCLES) // INTERVAL_SIZE
                        interval_counts[interval_idx] += 1
                else:
                    history[feature] = current_action

    except OSError:
        return None

    change_ratio = (total_changes / valid_cycles) if valid_cycles > 0 else 0.0

    return CsvRecord(
        benchmark=benchmark, model=model_name, size=size, freq=freq,
        total_changes=total_changes, valid_cycles=valid_cycles,
        change_ratio=change_ratio, interval_changes=dict(interval_counts)
    )

# =============================================================================
# 3. 结果输出与排版
# =============================================================================

def build_pivot_table(records: List[CsvRecord], attr_name: str, title: str, fmt: str) -> str:
    lines = []
    lines.append(f" {title} ".center(90, "="))
    
    grouped = defaultdict(list)
    for r in records: grouped[r.freq].append(r)
    sorted_freqs = sorted(grouped.keys())
    
    for freq in sorted_freqs:
        sub_records = grouped[freq]
        sorted_sizes = sorted(list({r.size for r in sub_records}))
        benchmarks = sorted({r.benchmark for r in sub_records})
        models = sorted(list({r.model for r in sub_records}), reverse=True) 
        
        table = {(r.benchmark, r.size, r.model): getattr(r, attr_name) for r in sub_records}

        bm_width = max(len("Benchmark"), max(len(b) for b in benchmarks)) + 2
        col_width = 14

        lines.append(f"\n========== DRAM Frequency: {freq} MHz ==========")
        
        header_size = " " * bm_width
        header_names = "Benchmark".ljust(bm_width)
        for sz in sorted_sizes:
            block_width = col_width * len(models) + 2
            header_size += f"|--- Size: {sz} ---".ljust(block_width)
            for m in models:
                header_names += m.rjust(col_width)
            header_names += " |"
            
        lines.append(header_size)
        lines.append(header_names)
        lines.append("-" * len(header_names))

        for bm in benchmarks:
            row = bm.ljust(bm_width)
            for sz in sorted_sizes:
                for m in models:
                    v = table.get((bm, sz, m))
                    cell = "-" if v is None else fmt.format(v)
                    row += cell.rjust(col_width)
                row += " |"
            lines.append(row)
    return "\n".join(lines)

def export_interval_details(records: List[CsvRecord], out_file: str):
    lines = []
    lines.append(" Interval Breakdown (Changes per 1M cycles) ".center(100, "="))
    lines.append("Interval Index: 0 means [10M - 11M], 1 means [11M - 12M], etc.")
    
    sorted_records = sorted(records, key=lambda r: (r.freq, r.benchmark, r.size, r.model))
    for r in sorted_records:
        lines.append(f"\n[{r.benchmark}] Freq: {r.freq} | Size: {r.size} | Model: {r.model}")
        lines.append(f"-> Total Changes: {r.total_changes:,} | Valid Instructions (cycles >= 10M): {r.valid_cycles:,}")
        
        if not r.interval_changes:
            lines.append("-> No changes recorded in intervals.")
            continue
        
        max_idx = max(r.interval_changes.keys())
        interval_str = ""
        for i in range(max_idx + 1):
            val = r.interval_changes.get(i, 0)
            interval_str += f"[{i}]: {val:<6}  "
            if (i + 1) % 10 == 0: 
                interval_str += "\n" + " " * 4
        lines.append("-> " + interval_str.strip())
    Path(out_file).write_text("\n".join(lines), encoding="utf-8")

# =============================================================================
# 4. 主函数与多进程调度
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Multi-processed CSV analyzer for ChampSim.")
    parser.add_argument("-d", "--dir", type=str, default=DEFAULT_DIR, help="Directory containing CSV files.")
    parser.add_argument("-w", "--workers", type=int, default=multiprocessing.cpu_count(), 
                        help="Number of parallel worker processes.")
    args = parser.parse_args()

    base_path = Path(args.dir).expanduser()
    if not base_path.exists():
        sys.exit(f"[error] Directory not found: {args.dir}")

    print(f"[info] Scanning directory: {base_path}")
    csv_files = [fp for fp in base_path.glob("**/*.csv") if "from_file" in fp.name or "pathfinder" in fp.name]
    total_files = len(csv_files)
    print(f"[info] Found {total_files} relevant CSV files.")
    print(f"[info] Starting processing with {args.workers} concurrent workers...")

    records = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process_csv, fp): fp for fp in csv_files}
        
        for i, future in enumerate(as_completed(futures), 1):
            try:
                record = future.result()
                if record:
                    records.append(record)
                print(f"\r[progress] Processing files... {i}/{total_files} ({(i/total_files)*100:.1f}%)", end="")
            except Exception as e:
                fp = futures[future]
                print(f"\n[error] Failed processing {fp.name}: {e}")
                
    print("\n[info] All files processed. Generating reports...")

    if not records:
        print("[warning] No valid records generated.")
        return

    txt_count = build_pivot_table(records, "total_changes", "Total Action Changes", "{:,}")
    Path("csv_metric_changes.txt").write_text(txt_count, encoding="utf-8")
    print("  -> Saved Total Changes to: csv_metric_changes.txt")
    
    txt_ratio = build_pivot_table(records, "change_ratio", "Change Ratio (Changes / Valid Instructions)", "{:.6f}")
    Path("csv_metric_ratio.txt").write_text(txt_ratio, encoding="utf-8")
    print("  -> Saved Change Ratio to: csv_metric_ratio.txt")
    
    export_interval_details(records, "csv_interval_breakdown.txt")
    print("  -> Saved Interval Breakdown to: csv_interval_breakdown.txt")

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()