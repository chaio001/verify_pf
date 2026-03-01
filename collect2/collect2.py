#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
import argparse
import json
import math
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

# =============================================================================
# 1. 核心数据结构与解析逻辑
# =============================================================================

RE_IPC          = re.compile(r"cumulative IPC:\s*([0-9]*\.?[0-9]+)", re.IGNORECASE)
RE_LLC_PREF_REQ = re.compile(r"LLC PREFETCH\s+REQUESTED:\s+\d+\s+ISSUED:\s+(\d+)\s+USEFUL:\s+(\d+)\s+USELESS:\s+(\d+)", re.IGNORECASE)
RE_LLC_LOAD     = re.compile(r"LLC LOAD\s+ACCESS:\s+(\d+)\s+HIT:\s+\d+\s+MISS:\s+(\d+)", re.IGNORECASE)
RE_PREF_FILL    = re.compile(r"PREFETCH\s+FILL:\s+\d+\s+LATE:\s+(\d+)\s+DROPPED:", re.IGNORECASE)
RE_LLC_PREF_ACC = re.compile(r"LLC PREFETCH\s+ACCESS:\s+\d+\s+HIT:\s+(\d+)\s+MISS:\s+(\d+)", re.IGNORECASE)
RE_LLC_TOTAL    = re.compile(r"LLC TOTAL\s+ACCESS:\s+\d+\s+HIT:\s+\d+\s+MISS:\s+(\d+)", re.IGNORECASE)

FREQ_SUFFIX_RE = re.compile(r"^(.*)_(\d+)(\.txt)$")

@dataclass(frozen=True)
class Record:
    name: str
    benchmark: str
    freq: str
    file: str
    
    ipc: float
    accuracy: float
    coverage: float
    bw_share: float
    
    issued: int
    useful: int
    useless: int
    late: int
    hit: int
    others: int

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
        fake_filename = m.group(1) + m.group(3)
        freq = m.group(2)
        return fake_filename, freq
    return filename, "unknown"

def pick_rule(fp: Path, base_filename: str, rules: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    virtual_path = fp.parent / base_filename
    full_path_str = str(fp)
    for rule in rules:
        pc = rule.get("path_contains")
        if pc and pc not in full_path_str:
            continue
        file_globs = rule.get("file_globs", [])
        if not file_globs:
            return rule
        if any(virtual_path.match(g) for g in file_globs):
            return rule
    return None

def infer_benchmark(filename: str, strip_suffixes: List[str]) -> str:
    for sfx in sorted(strip_suffixes or [], key=len, reverse=True):
        if filename.endswith(sfx):
            return filename[: -len(sfx)]
    return Path(filename).stem

def extract_metrics_from_file(fp: Path) -> Optional[Dict[str, Any]]:
    data = {
        'ipc': 0.0, 'issued': 0, 'useful': 0, 'useless': 0,
        'load_access': 0, 'late': 0, 'pref_hit': 0, 'pref_miss': 0,
        'total_miss': 0, 'found_any': False
    }
    try:
        with fp.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                m = RE_IPC.search(line)
                if m: data['ipc'] = float(m.group(1)); data['found_any'] = True
                
                m = RE_LLC_PREF_REQ.search(line)
                if m:
                    data['issued'] = int(m.group(1))
                    data['useful'] = int(m.group(2))
                    data['useless'] = int(m.group(3))
                    
                m = RE_LLC_LOAD.search(line)
                if m: data['load_access'] = int(m.group(1))
                
                m = RE_PREF_FILL.search(line)
                if m: data['late'] = int(m.group(1))
                
                m = RE_LLC_PREF_ACC.search(line)
                if m:
                    data['pref_hit'] = int(m.group(1))
                    data['pref_miss'] = int(m.group(2))
                    
                m = RE_LLC_TOTAL.search(line)
                if m: data['total_miss'] = int(m.group(1))
    except OSError:
        return None
        
    if not data['found_any']:
        return None
    return data

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
        benchmark = infer_benchmark(base_filename, rule.get("strip_suffixes", []))

        raw = extract_metrics_from_file(fp)
        if not raw:
            continue

        issued, useful, useless = raw['issued'], raw['useful'], raw['useless']
        load_access, late = raw['load_access'], raw['late']
        pref_hit, pref_miss, total_miss = raw['pref_hit'], raw['pref_miss'], raw['total_miss']

        accuracy = useful / issued if issued > 0 else 0.0
        coverage = useful / load_access if load_access > 0 else 0.0
        bw_share = pref_miss / total_miss if total_miss > 0 else 0.0
        others = issued - useful - useless - late - pref_hit

        records.append(Record(
            name=name, benchmark=benchmark, freq=freq, file=str(fp),
            ipc=raw['ipc'], accuracy=accuracy, coverage=coverage, bw_share=bw_share,
            issued=issued, useful=useful, useless=useless, late=late, hit=pref_hit, others=others
        ))
    return records

# =============================================================================
# 2. 纯文本二维表格生成与多文件输出模块
# =============================================================================

def get_size_and_prefix(name: str):
    m = re.search(r"(\d+)$", name)
    if m:
        size = m.group(1)
        return int(size), name[:-len(size)]
    return 9999, name

def build_pivot_table_lines(records: List[Record], attr_name: str, metric_title: str, fmt: str) -> List[str]:
    lines = []
    lines.append(f" {metric_title} ".center(80, "="))
    
    grouped = defaultdict(list)
    for r in records:
        grouped[r.freq].append(r)

    sorted_freqs = sorted(grouped.keys(), key=lambda x: int(x) if x.isdigit() else x)
    
    for freq in sorted_freqs:
        sub_records = grouped[freq]
        all_names = sorted(list({r.name for r in sub_records}), key=get_size_and_prefix)
        
        size_groups = defaultdict(list)
        for n in all_names:
            sz, _ = get_size_and_prefix(n)
            size_groups[sz].append(n)
        
        sorted_sizes = sorted(size_groups.keys())
        benchmarks = sorted({r.benchmark for r in sub_records})
        table = {(r.benchmark, r.name): getattr(r, attr_name) for r in sub_records}

        bm_width = max(len("Benchmark"), max(len(b) for b in benchmarks)) + 2
        col_width = 16 # 拓宽一点以容纳 '>>'

        lines.append(f"\n========== DRAM Frequency: {freq} MHz ==========")
        
        header_size = " " * bm_width
        header_names = "Benchmark".ljust(bm_width)
        
        for sz in sorted_sizes:
            group_names = size_groups[sz]
            # 【对齐修复】严格计算底下模型列占据的总宽度，+2 是为了包含 " |" 分隔符
            block_width = col_width * len(group_names) + 2
            
            # 使用 ljust 严格填充空格，保证上下宽度100%一致
            header_size += f"|--- Size: {sz} ---".ljust(block_width)
            
            for n in group_names:
                header_names += n.replace(str(sz), "").rjust(col_width)
            header_names += " |"
            
        lines.append(header_size)
        lines.append(header_names)
        lines.append("-" * len(header_names))

        for bm in benchmarks:
            row = bm.ljust(bm_width)
            for sz in sorted_sizes:
                group_names = size_groups[sz]
                vals = [table.get((bm, n)) for n in group_names if table.get((bm, n)) is not None]
                local_max = max(vals) if vals else None

                for n in group_names:
                    v = table.get((bm, n))
                    if v is None:
                        cell = "-"
                    else:
                        cell = fmt.format(v)
                        # 给最大值加上 '>>' (排除 0 值)
                        if local_max is not None and abs(v - local_max) < 1e-9 and local_max > 0:
                            cell = ">>" + cell
                    row += cell.rjust(col_width)
                row += " |"
            lines.append(row)
    return lines

# ...(此处省略 build_prefetch_breakdown_lines 和 build_summary_table_lines 的代码，与上一版完全一致)...
def build_prefetch_breakdown_lines(records: List[Record]) -> List[str]:
    lines = []
    lines.append(" Useful/Useless Prefetch Breakdown ".center(120, "="))
    grouped = defaultdict(lambda: defaultdict(list))
    for r in records: grouped[r.freq][r.benchmark].append(r)
    sorted_freqs = sorted(grouped.keys(), key=lambda x: int(x) if x.isdigit() else x)
    metrics = ["Issued", "Useful", "Useless", "Late", "Hit", "Others"]

    for freq in sorted_freqs:
        lines.append(f"\n========== DRAM Frequency: {freq} MHz ==========")
        for bm in sorted(grouped[freq].keys()):
            sub_records = grouped[freq][bm]
            all_names = sorted(list({r.name for r in sub_records}), key=get_size_and_prefix)
            size_groups = defaultdict(list)
            for n in all_names:
                sz, _ = get_size_and_prefix(n)
                size_groups[sz].append(n)
            
            sorted_sizes = sorted(size_groups.keys())
            col_width = 12
            lines.append(f"\n[ Benchmark: {bm} ]")

            header_size = "Metric".ljust(10)
            header_names = "".ljust(10)

            for sz in sorted_sizes:
                group_names = size_groups[sz]
                # 【对齐修复】严格计算宽度
                block_width = col_width * len(group_names) + 2
                
                header_size += f"|--- Size: {sz} ---".ljust(block_width)
                
                for n in group_names:
                    header_names += n.replace(str(sz), "").rjust(col_width)
                header_names += " |"
                
            lines.append(header_size)
            lines.append(header_names)
            lines.append("-" * len(header_names))

            rec_dict = {r.name: r for r in sub_records}
            for metric in metrics:
                row = metric.ljust(10)
                for sz in sorted_sizes:
                    for n in size_groups[sz]:
                        rec = rec_dict.get(n)
                        if rec:
                            val = getattr(rec, metric.lower())
                            cell = f"{val:,}" if val != 0 else "0"
                        else:
                            cell = "-"
                        row += cell.rjust(col_width)
                    row += " |"
                lines.append(row)
    return lines

def build_summary_table_lines(records: List[Record]) -> List[str]:
    lines = []
    lines.append(" Global Summary Table ".center(120, "="))
    bm_width = max([len(r.benchmark) for r in records] + [10]) + 2
    name_width = max([len(r.name) for r in records] + [10]) + 2
    header = (f"{'Benchmark'.ljust(bm_width)}{'Model'.ljust(name_width)}{'Freq'.ljust(6)}"
              f"{'IPC'.rjust(8)}{'Accuracy'.rjust(10)}{'Coverage'.rjust(10)}{'BW_Share'.rjust(10)} | "
              f"{'Issued'.rjust(8)}{'Useful'.rjust(8)}{'Useless'.rjust(8)}{'Late'.rjust(8)}"
              f"{'Hit'.rjust(8)}{'Others'.rjust(8)}")
    lines.append(header)
    lines.append("-" * len(header))
    sorted_records = sorted(records, key=lambda r: (r.benchmark, int(r.freq) if r.freq.isdigit() else 0, get_size_and_prefix(r.name)))
    for r in sorted_records:
        row = (f"{r.benchmark.ljust(bm_width)}{r.name.ljust(name_width)}{r.freq.ljust(6)}"
               f"{r.ipc:>8.4f}{r.accuracy:>10.2%}{r.coverage:>10.2%}{r.bw_share:>10.2%} | "
               f"{r.issued:>8}{r.useful:>8}{r.useless:>8}{r.late:>8}{r.hit:>8}{r.others:>8}")
        lines.append(row)
    return lines

def write_files(records: List[Record]) -> None:
    if not records:
        print("No records found to display.")
        return

    metric_configs = [
        ("ipc", "Cumulative IPC", "{:.5f}", "metric_ipc.txt"),
        ("accuracy", "Prefetch Accuracy", "{:.2%}", "metric_accuracy.txt"),
        ("coverage", "Prefetch Coverage", "{:.2%}", "metric_coverage.txt"),
        ("bw_share", "Prefetch Bandwidth Share", "{:.2%}", "metric_bw_share.txt")
    ]
    print("\n[info] Generating individual metric files...")
    for attr, title, fmt, filename in metric_configs:
        lines = build_pivot_table_lines(records, attr, title, fmt)
        Path(filename).write_text("\n".join(lines), encoding="utf-8")
        print(f"  -> Saved {title} to: {filename}")

    breakdown_lines = build_prefetch_breakdown_lines(records)
    Path("metric_breakdown.txt").write_text("\n".join(breakdown_lines), encoding="utf-8")
    print(f"  -> Saved Breakdown to: metric_breakdown.txt")

    summary_lines = build_summary_table_lines(records)
    Path("metric_summary.txt").write_text("\n".join(summary_lines), encoding="utf-8")
    print(f"  -> Saved Summary to: metric_summary.txt")


# =============================================================================
# 3. 增强可视化：Breakdown 高级堆叠分组柱状图
# =============================================================================

def plot_prefetch_breakdown(records: List[Record]):
    """按 Size 生成对应的 Breakdown 图像，X轴为频率，不同模型并排显示"""
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        import pandas as pd
        import numpy as np
    except ImportError:
        print("\n[warning] 缺少 matplotlib 或 pandas，跳过 Breakdown 绘图。")
        return

    print("\n[info] 正在生成 Breakdown 组装堆叠柱状图...")

    data = []
    for r in records:
        size, base_model = get_size_and_prefix(r.name)
        if base_model == "no": continue  # 过滤掉 baseline

        # 处理频率为纯数字以便正确排序
        freq_val = int(r.freq) if r.freq.isdigit() else r.freq
        data.append({
            "Freq": freq_val, "Benchmark": r.benchmark,
            "Model": base_model, "Size": size,
            "Useful": r.useful, "Useless": r.useless,
            "Late": r.late, "Hit": r.hit, "Others": r.others
        })

    if not data: return
    df = pd.DataFrame(data)
    components = ["Useful", "Useless", "Late", "Hit", "Others"]
    
    # 严格按照你提供的图片风格：颜色和斜线网纹 (Hatch)
    colors = ["#4daf4a", "#e41a1c", "#ffffff", "#a0a0a0", "#e0e0e0"]
    hatches = ["", "//", "//", "//", ""]
    legend_labels = [
        "Useful (prefetch used)", "Useless (prefetch not used)", 
        "Late (prefetch untimely)", "hit (prefetch hit by LLC)", "others (others)"
    ]

    # 按硬件 Size 分开生成图片 (比如 2048 一张图，4096 一张图)
    for size, df_size in df.groupby("Size"):
        freqs = sorted(df_size['Freq'].unique(), key=lambda x: int(x) if isinstance(x, str) and x.isdigit() else x)
        benchmarks = sorted(df_size['Benchmark'].unique())
        models = sorted(df_size['Model'].unique()) # 通常是 ['pathfinder', 'pythia']
        
        num_models = len(models)
        ncols = 4 if len(benchmarks) >= 4 else len(benchmarks)
        nrows = math.ceil(len(benchmarks) / ncols) if ncols > 0 else 1
        
        # 建立网格画布
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(4 * ncols, 3.5 * nrows), squeeze=False)
        fig.subplots_adjust(top=0.78, hspace=0.4, wspace=0.25)

        
        # 动态标题
        subtitle_bars = " | ".join([f"{['Left', 'Right', 'Middle'][i % 3]} Bar: {m.capitalize()}" for i, m in enumerate(models)])
        fig.suptitle(f"Dynamic Prefetch Adjustment by Freq @ Size={size}\n({subtitle_bars})", 
                     fontsize=15, fontweight='bold', y=0.98)
        
        axes_flat = axes.flatten()
        
        for idx, bm in enumerate(benchmarks):
            ax = axes_flat[idx]
            df_bm = df_size[df_size['Benchmark'] == bm]
            
            x = np.arange(len(freqs))
            total_width = 0.8
            bar_width = total_width / num_models
            
            # 遍历每个频率
            for f_idx, freq in enumerate(freqs):
                df_f = df_bm[df_bm['Freq'] == freq]
                
                # 遍历同一个频率下的不同预取器 (Pythia, Pathfinder并排排列)
                for m_idx, model in enumerate(models):
                    m_row = df_f[df_f['Model'] == model]
                    m_vals = [m_row[c].values[0] if not m_row.empty else 0 for c in components]
                    
                    bottom = 0
                    x_offset = (m_idx - num_models / 2.0 + 0.5) * bar_width
                    
                    # 向上堆叠 5 种成分
                    for c_idx, comp in enumerate(components):
                        ax.bar(x[f_idx] + x_offset, m_vals[c_idx], width=bar_width, bottom=bottom, 
                               color=colors[c_idx], hatch=hatches[c_idx], edgecolor='black', linewidth=0.8)
                        bottom += m_vals[c_idx]
            
            ax.set_xticks(x)
            ax.set_xticklabels([str(f) for f in freqs], rotation=45)
            ax.set_title(bm, fontsize=11, pad=5)
            if idx % ncols == 0:
                ax.set_ylabel("Absolute Prefetch Count", fontsize=9)
            ax.grid(axis='y', linestyle='--', alpha=0.6)
            ax.set_axisbelow(True)
            
        # 隐藏多余的空白子图
        for idx in range(len(benchmarks), len(axes_flat)):
            axes_flat[idx].set_visible(False)
            
        # 顶部全局图例
        legend_patches = [mpatches.Patch(facecolor=colors[i], hatch=hatches[i], edgecolor='black', label=legend_labels[i]) for i in range(len(components))]
        fig.legend(handles=legend_patches, loc='upper center', bbox_to_anchor=(0.5, 0.90), ncol=3, frameon=False, fontsize=10)
        # 保存图片
        out_name = f"plot_breakdown_size{size}.png"
        plt.savefig(out_name, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"  -> Saved grouped breakdown plot to: {out_name}")

# =============================================================================
# 4. 主函数入口
# =============================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to JSON configuration file")
    args = ap.parse_args()
    try:
        with open(args.config, "r") as f:
            cfg = json.load(f)
    except Exception as e:
        sys.exit(f"Error loading config: {e}")

    records = collect_records(cfg)
    write_files(records)
    plot_prefetch_breakdown(records)
    print(f"\n[info] Done! Total records collected: {len(records)}", file=sys.stderr)

if __name__ == "__main__":
    main()