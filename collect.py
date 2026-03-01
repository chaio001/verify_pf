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

# =============================================================================
# 1. 核心数据结构与解析逻辑
# =============================================================================

FINISHED_IPC_RE = re.compile(
    r"Finished\s+CPU\s+0\s+instructions:.*?cumulative\s+IPC:\s*([0-9]*\.?[0-9]+)",
    re.IGNORECASE,
)

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
            print(f"[debug] No IPC found in: {fp.name}", file=sys.stderr)
            continue

        line_no, ipc = ipcs[-1]
        records.append(Record(name=name, benchmark=benchmark, ipc=ipc, freq=freq, file=str(fp), line_no=line_no))

    return records

# =============================================================================
# 2. 纯文本表格输出模块 
# =============================================================================

def write_txt_pivot_grouped(records: List[Record], out_file: Optional[str]) -> None:
    if not records:
        print("No records found to display.")
        return

    grouped = defaultdict(list)
    for r in records:
        grouped[r.freq].append(r)

    sorted_freqs = sorted(grouped.keys(), key=lambda x: int(x) if x.isdigit() else x)
    
    def get_size_and_prefix(name: str):
        m = re.search(r"(\d+)$", name)
        if m:
            size = m.group(1)
            prefix = name[:-len(size)]
            return int(size), prefix
        return 9999, name

    final_lines = []

    for freq in sorted_freqs:
        sub_records = grouped[freq]
        all_names = sorted(list({r.name for r in sub_records}), key=get_size_and_prefix)
        
        size_groups = defaultdict(list)
        for n in all_names:
            sz, _ = get_size_and_prefix(n)
            size_groups[sz].append(n)
        
        sorted_sizes = sorted(size_groups.keys())
        benchmarks = sorted({r.benchmark for r in sub_records})
        table = {(r.benchmark, r.name): r.ipc for r in sub_records}

        bm_width = max(len("Benchmark"), max(len(b) for b in benchmarks)) + 2
        col_width = 12

        final_lines.append(f"\n========== DRAM Frequency: {freq} MHz ==========")
        
        header_size = " " * bm_width
        header_names = "Benchmark".ljust(bm_width)
        
        for sz in sorted_sizes:
            header_size += f"|--- Size: {sz} ---".center(col_width * len(size_groups[sz])).rstrip() + " "
            for n in size_groups[sz]:
                pure_name = n.replace(str(sz), "")
                header_names += pure_name.rjust(col_width)
            header_names += " |"
        
        final_lines.append(header_size)
        final_lines.append(header_names)
        final_lines.append("-" * len(header_names))

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
                        cell = f"{v:.6f}"
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

# =============================================================================
# 3. 增强可视化模块 (以Baseline加速比为Y轴)
# =============================================================================

def generate_visualizations(records: List[Record]):
    
    model_palette = {
        "no": "#4daf4a",          # 绿色 baseline
        "pythia": "#e41a1c",      # 红
        "pathfinder": "#377eb8",  # 蓝
    }

    try:
        import pandas as pd
        import seaborn as sns
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n[warning] 缺少绘图依赖，跳过可视化生成。")
        print("如果需要图表，请运行: pip install pandas seaborn matplotlib")
        return

    print("\n[info] 正在生成可视化图表...")
    
    data = []
    for r in records:
        m = re.search(r"(\d+)$", r.name)
        if m:
            size = int(m.group(1))
            model = r.name[:-len(m.group(1))]
        else:
            continue
            
        data.append({
            "Benchmark": r.benchmark,
            "Freq": int(r.freq) if r.freq.isdigit() else 0,
            "Size": size,
            "Model": model,
            "IPC": r.ipc
        })
    
    df = pd.DataFrame(data)
    if df.empty:
        print("[warning] 没有足够的数据用于生成图表。")
        return

# -------------------------------------------------------------------------
    # 数据转换：将绝对 IPC 转换为相对于 'no' (Baseline) 的加速比
    # -------------------------------------------------------------------------
    pivot_df = df.pivot_table(index=["Benchmark", "Freq", "Size"], columns="Model", values="IPC").reset_index()
    
    if 'no' not in pivot_df.columns:
        print("[warning] 数据集中找不到 baseline ('no')，无法计算加速比。放弃生成图表。")
        return
        
    models = [col for col in pivot_df.columns if col not in ["Benchmark", "Freq", "Size"]]
    
    # 【修复重点】：先克隆一份 baseline 数据作为除数，防止它在循环中被原地覆盖成 1.0
    baseline_ipc = pivot_df['no'].copy()
    
    # 统一计算 Speedup (Ratio)
    for m in models:
        pivot_df[m] = pivot_df[m] / baseline_ipc
        
    # 将宽表融化回长表，以便 Seaborn 绘图
    speedup_df = pivot_df.melt(id_vars=["Benchmark", "Freq", "Size"], value_vars=models, var_name="Model", value_name="Speedup")


    # --- 图 1：选择最高频率，绘制 Speedup - Size 扩展趋势折线图 ---
    # max_freq = speedup_df['Freq'].max()
    max_freq = 2400
    df_max_freq = speedup_df[speedup_df['Freq'] == max_freq]
    
    if not df_max_freq.empty:
        sns.set_theme(style="whitegrid")
        g = sns.relplot(
            data=df_max_freq, x="Size", y="Speedup", hue="Model", 
            style="Model", markers=True, dashes=False,
            col="Benchmark", col_wrap=4, kind="line", 
            height=3, aspect=1.2, palette=model_palette#"Set1"
        )
        g.fig.suptitle(f"Speedup vs Size @ {max_freq} MHz (Baseline = 'no')", y=1.05, fontsize=16, fontweight='bold')
        
        for ax in g.axes.flat:
            ax.set_xscale('log', base=2)
            sizes = sorted(df_max_freq['Size'].unique())
            ax.set_xticks(sizes)
            ax.set_xticklabels(sizes)
            
            # 【重要】画出 y=1.0 的基准线
            ax.axhline(1.0, color='black', linestyle='--', alpha=0.5, zorder=0)
            
        plt.savefig(f"plot_speedup_scaling_{max_freq}MHz.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"[info] 已生成折线图: plot_speedup_scaling_{max_freq}MHz.png")


    # --- 图 2：固定 Size=2048，绘制 Speedup - Freq 扩展趋势折线图 ---
    df_2048 = speedup_df[speedup_df['Size'] == 2048]
    
    if not df_2048.empty:
        sns.set_theme(style="whitegrid")
        g2 = sns.relplot(
            data=df_2048, x="Freq", y="Speedup", hue="Model", 
            style="Model", markers=True, dashes=False,
            col="Benchmark", col_wrap=4, kind="line", 
            height=3, aspect=1.2, palette=model_palette#"Set2"
        )
        g2.fig.suptitle("Speedup vs Freq @ Size 2048 (Baseline = 'no')", y=1.05, fontsize=16, fontweight='bold')
        
        for ax in g2.axes.flat:
            ax.set_xscale('log', base=2)
            freqs = sorted(df_2048['Freq'].unique())
            ax.set_xticks(freqs)
            ax.set_xticklabels(freqs)
            ax.tick_params(axis='x', rotation=45)
            
            # 【重要】画出 y=1.0 的基准线
            ax.axhline(1.0, color='black', linestyle='--', alpha=0.5, zorder=0)
            
        plt.savefig("plot_speedup_scaling_size2048.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"[info] 已生成折线图: plot_speedup_scaling_size2048.png")


    # --- 图 3：Pythia 和 Pathfinder 相对于 Baseline 的整体平均加速比热力图 ---
    # 这里依然沿用百分比 (Pct) 展示，因为热力图上写 "+5.2%" 比写 "1.05" 更清晰
    models_to_compare = [m for m in ['pythia', 'pathfinder'] if m in pivot_df.columns]
    
    for model in models_to_compare:
        # Pct = (Speedup - 1) * 100
        pivot_df[f'{model}_Speedup_Pct'] = (pivot_df[model] - 1) * 100
        heatmap_data = pivot_df.pivot_table(index="Freq", columns="Size", values=f"{model}_Speedup_Pct", aggfunc="mean")
        heatmap_data = heatmap_data.sort_index(ascending=False)

        plt.figure(figsize=(8, 5))
        sns.heatmap(
            heatmap_data, annot=True, fmt=".2f", cmap="coolwarm", 
            center=0, cbar_kws={'label': f'Avg Speedup over Baseline (%)'}
        )
        plt.title(f"{model.capitalize()} Pre-fetcher Overall Speedup", pad=15, fontweight='bold')
        plt.ylabel("DRAM Frequency (MHz)")
        plt.xlabel("Hardware Size")
        
        out_name = f"plot_{model}_speedup_heatmap.png"
        plt.savefig(out_name, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"[info] 已生成热力图: {out_name}")


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
    
    # 1. 打印/保存文本表格
    write_txt_pivot_grouped(records, cfg.get("output"))
    print(f"\n[info] Total records collected: {len(records)}", file=sys.stderr)
    
    # 2. 生成可视化图表
    generate_visualizations(records)

if __name__ == "__main__":
    main()