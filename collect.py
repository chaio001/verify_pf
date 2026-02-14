#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
import argparse
import json
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

# IPC 匹配正则
FINISHED_IPC_RE = re.compile(
    r"Finished\s+CPU\s+0\s+instructions:.*?cumulative\s+IPC:\s*([0-9]*\.?[0-9]+)",
    re.IGNORECASE,
)

# 频率后缀匹配：提取文件名末尾的 _频率.txt
FREQ_SUFFIX_RE = re.compile(r"^(.*)_(\d+)(\.txt)$")

@dataclass(frozen=True)
class Record:
    name: str
    benchmark: str
    ipc: float
    freq: str
    file: str
    line_no: int

def iter_target_files(paths: List[str], include_globs: List[str]) -> Iterable[Path]:
    for p in paths:
        base = Path(p).expanduser()
        if not base.exists():
            continue
        for g in include_globs:
            yield from base.glob(g)

def parse_freq_and_base(filename: str) -> Tuple[str, str]:
    m = FREQ_SUFFIX_RE.match(filename)
    if m:
        # 去掉频率部分，还原成 base_filename 用于匹配 JSON 规则
        fake_filename = m.group(1) + m.group(3)
        freq = m.group(2)
        return fake_filename, freq
    return filename, "unknown"

def pick_rule(fp: Path, base_filename: str, rules: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    # 【关键修复】：构造虚拟路径，保留目录结构，只替换文件名为剥离频率后的版本
    # 这样 JSON 里的 **/* 匹配才能生效
    virtual_path = fp.parent / base_filename
    full_path_str = str(fp)
    
    for rule in rules:
        pc = rule.get("path_contains")
        if pc and pc not in full_path_str:
            continue

        file_globs = rule.get("file_globs", [])
        if not file_globs:
            return rule

        # 使用虚拟路径进行匹配
        if any(virtual_path.match(g) for g in file_globs):
            return rule
    return None

def infer_benchmark(filename: str, strip_suffixes: List[str]) -> str:
    # 最长匹配优先
    for sfx in sorted(strip_suffixes or [], key=len, reverse=True):
        if filename.endswith(sfx):
            return filename[: -len(sfx)]
    return Path(filename).stem

def extract_ipcs_from_file(fp: Path) -> List[Tuple[int, float]]:
    out: List[Tuple[int, float]] = []
    try:
        with fp.open("r", encoding="utf-8", errors="ignore") as f:
            for i, line in enumerate(f, start=1):
                m = FINISHED_IPC_RE.search(line)
                if m:
                    try:
                        out.append((i, float(m.group(1))))
                    except ValueError:
                        pass
    except OSError:
        return []
    return out

def collect_records(cfg: Dict[str, Any]) -> List[Record]:
    paths: List[str] = cfg["paths"]
    include_globs: List[str] = cfg.get("include_globs", ["**/*.txt"])
    rules: List[Dict[str, Any]] = cfg.get("rules", [])

    records: List[Record] = []

    for fp in iter_target_files(paths, include_globs):
        if not fp.is_file():
            continue

        base_filename, freq = parse_freq_and_base(fp.name)
        rule = pick_rule(fp, base_filename, rules)
        
        if rule is None:
            continue 

        name = rule.get("name", "unknown")
        strip_suffixes = rule.get("strip_suffixes", [])
        benchmark = infer_benchmark(base_filename, strip_suffixes)

        ipcs = extract_ipcs_from_file(fp)
        if not ipcs:
            # 如果没采集到 IPC，打印一个警告到 stderr 方便你排查是不是文件还没跑完
            print(f"[debug] No IPC found in: {fp.name}", file=sys.stderr)
            continue

        line_no, ipc = ipcs[-1]
        records.append(Record(name=name, benchmark=benchmark, ipc=ipc, freq=freq, file=str(fp), line_no=line_no))

    return records

def write_txt_pivot_grouped(records: List[Record], out_file: Optional[str]) -> None:
    if not records:
        print("No records found to display.")
        return

    # 按频率分组
    grouped = defaultdict(list)
    for r in records:
        grouped[r.freq].append(r)

    sorted_freqs = sorted(grouped.keys(), key=lambda x: int(x) if x.isdigit() else x)
    
    # 定义前缀顺序，用于小模块内部排序
    prefix_order = ["no", "pathfinder", "pythia"]

    def get_size_and_prefix(name: str):
        # 从 "pathfinder1024" 提取 ("1024", "pathfinder")
        m = re.search(r"(\d+)$", name)
        if m:
            size = m.group(1)
            prefix = name[:-len(size)]
            return int(size), prefix
        return 9999, name

    final_lines = []

    for freq in sorted_freqs:
        sub_records = grouped[freq]
        
        # 获取所有出现的名称，并按 (Size, Prefix_Order) 排序
        all_names = sorted(list({r.name for r in sub_records}), key=get_size_and_prefix)
        
        # 按 Size 将名称分组，例如: {256: ["no256", "pathfinder256", "pythia256"], 1024: [...]}
        size_groups = defaultdict(list)
        for n in all_names:
            sz, _ = get_size_and_prefix(n)
            size_groups[sz].append(n)
        
        sorted_sizes = sorted(size_groups.keys())
        benchmarks = sorted({r.benchmark for r in sub_records})
        table = {(r.benchmark, r.name): r.ipc for r in sub_records}

        # 计算宽度
        bm_width = max(len("Benchmark"), max(len(b) for b in benchmarks)) + 2
        col_width = 12

        final_lines.append(f"\n========== DRAM Frequency: {freq} MHz ==========")
        
        # 构建表头 (两行，第一行显示 Size)
        header_size = " " * bm_width
        header_names = "Benchmark".ljust(bm_width)
        
        for sz in sorted_sizes:
            header_size += f"|--- Size: {sz} ---".center(col_width * len(size_groups[sz])).rstrip() + " "
            for n in size_groups[sz]:
                # 简化显示，只显示前缀
                pure_name = n.replace(str(sz), "")
                header_names += pure_name.rjust(col_width)
            header_names += " |" # 竖线分隔
        
        final_lines.append(header_size)
        final_lines.append(header_names)
        final_lines.append("-" * len(header_names))

        # 构建行数据
        for bm in benchmarks:
            row = bm.ljust(bm_width)
            for sz in sorted_sizes:
                group_names = size_groups[sz]
                # 局部对比：只在当前 Size 组内找最大值
                vals = [table.get((bm, n)) for n in group_names if table.get((bm, n)) is not None]
                local_max = max(vals) if vals else None

                for n in group_names:
                    v = table.get((bm, n))
                    if v is None:
                        cell = "-"
                    else:
                        cell = f"{v:.4f}"
                        if local_max and abs(v - local_max) < 1e-9:
                            cell = ">>" + cell
                    row += cell.rjust(col_width)
                row += " |"
            final_lines.append(row)

    text = "\n".join(final_lines)
    if out_file:
        Path(out_file).write_text(text, encoding="utf-8")
        print(f"[info] Enhanced table written to {out_file}")
    else:
        print(text)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    try:
        with open(args.config, "r") as f:
            cfg = json.load(f)
    except Exception as e:
        sys.exit(f"Error loading config: {e}")

    records = collect_records(cfg)
    write_txt_pivot_grouped(records, cfg.get("output"))
    print(f"[info] Total records collected: {len(records)}", file=sys.stderr)

if __name__ == "__main__":
    main()