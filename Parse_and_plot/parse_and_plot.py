#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import csv
import glob
import argparse
import sys
import math

# ==========================================
# 默认配置区域
# ==========================================
DEFAULT_DIR = r"/mnt/sda/cyhcpp/Pythia/a_pythia/ChampSim-master/results"  # <-- 替换为你的真实路径
# DEFAULT_DIR = r"/mnt/sda/cyhcpp/Pythia/a_pythia/ChampSim-master/results-pythia_deltaseq"  # <-- 替换为你的真实路径

def parse_champsim_log(filepath):
    """解析单个 ChampSim 日志文件，返回提取的指标字典"""
    filename = os.path.basename(filepath)
    
    benchmark = filename.split('-hashed')[0].replace('.txt', '').replace('.log', '')
    size, freq = "0", "0"
    model = "unknown"
    
    m_sf = re.search(r'_(\d+)_(\d+)\.(txt|log|csv)$', filename)
    if m_sf:
        size, freq = m_sf.group(1), m_sf.group(2)
        benchmark = filename[:m_sf.start()].split('-hashed')[0] 
        
    if 'pathfinder' in filename:
        model = 'pathfinder'
    elif 'pythia' in filename or 'from_file' in filename:
        model = 'pythia'
    elif '-no-' in filename or 'no-lru' in filename:
        model = 'no'

    metrics = {
        'Trace_Name': filename,
        'Benchmark': benchmark,
        'Model': model,
        'Size': int(size) if size.isdigit() else 0,
        'Freq': int(freq) if freq.isdigit() else 0,
        'IPC': '0', 'Instructions': '0', 'Cycles': '0',
        'L1D_Access': '0', 'L1D_Hit': '0', 'L1D_Miss': '0',
        'L2C_Access': '0', 'L2C_Hit': '0', 'L2C_Miss': '0',
        'LLC_Access': '0', 'LLC_Hit': '0', 'LLC_Miss': '0',
        'LLC_Pref_Requested': '0', 'LLC_Pref_Issued': '0', 
        'LLC_Pref_Useful': '0', 'LLC_Pref_Useless': '0',
        'LLC_Pref_Late': '0',  

        'LLC_Pref_Hit': '0',  
        'LLC_Pref_OTHERS': '0',  
        'LLC_Pref_FILL': '0',  
        'LLC_Pref_DROPPED': '0',  
        'LLC_Pref_ACCESS': '0',  
        'LLC_Pref_MISS': '0',  

        'Avg_Congested_Cycle': '0'
    }

    p_ipc = re.compile(r"CPU 0 cumulative IPC:\s+([\d.]+)\s+instructions:\s+(\d+)\s+cycles:\s+(\d+)")
    p_l1d = re.compile(r"L1D TOTAL\s+ACCESS:\s+(\d+)\s+HIT:\s+(\d+)\s+MISS:\s+(\d+)")
    p_l2c = re.compile(r"L2C TOTAL\s+ACCESS:\s+(\d+)\s+HIT:\s+(\d+)\s+MISS:\s+(\d+)")
    p_llc = re.compile(r"LLC TOTAL\s+ACCESS:\s+(\d+)\s+HIT:\s+(\d+)\s+MISS:\s+(\d+)")
    p_llc_pref = re.compile(r"LLC PREFETCH\s+REQUESTED:\s+(\d+)\s+ISSUED:\s+(\d+)\s+USEFUL:\s+(\d+)\s+USELESS:\s+(\d+)")

    p_llc_pref_hm = re.compile(r"LLC PREFETCH\s+ACCESS:\s+(\d+)\s+HIT:\s+(\d+)\s+MISS:\s+(\d+)") # hit or miss
    p_llc_pref_oth = re.compile(r"FILL:\s+(\d+)\s+LATE:\s+(\d+)\s+DROPPED:\s+(\d+)")
    # p_llc_pref_oth = re.compile(r"LLC PREFETCH\s+FILL:\s+(\d+)\s+LATE:\s+(\d+)\s+DROPPED:\s+(\d+)")

    p_congest = re.compile(r"AVG_CONGESTED_CYCLE:\s+(\d+)")

    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                if m := p_ipc.search(line):
                    metrics['IPC'], metrics['Instructions'], metrics['Cycles'] = m.groups()
                elif m := p_llc.search(line):
                    metrics['LLC_Access'], metrics['LLC_Hit'], metrics['LLC_Miss'] = m.groups()
                elif m := p_llc_pref.search(line):
                    metrics['LLC_Pref_Requested'], metrics['LLC_Pref_Issued'], metrics['LLC_Pref_Useful'], metrics['LLC_Pref_Useless'] = m.groups()

                elif m := p_llc_pref_hm.search(line):
                    metrics['LLC_Pref_ACCESS'], metrics['LLC_Pref_Hit'], metrics['LLC_Pref_MISS'] = m.groups()
                elif m := p_llc_pref_oth.search(line):
                    metrics['LLC_Pref_FILL'], metrics['LLC_Pref_Late'], metrics['LLC_Pref_DROPPED'] = m.groups()

                elif m := p_l2c.search(line):
                    metrics['L2C_Access'], metrics['L2C_Hit'], metrics['L2C_Miss'] = m.groups()
                elif m := p_l1d.search(line):
                    metrics['L1D_Access'], metrics['L1D_Hit'], metrics['L1D_Miss'] = m.groups()
                elif m := p_congest.search(line):
                    metrics['Avg_Congested_Cycle'] = m.group(1)
                    
        # 计算 Late Prefetch
        issued = int(metrics['LLC_Pref_Issued'])
        useful = int(metrics['LLC_Pref_Useful'])
        useless = int(metrics['LLC_Pref_Useless'])
        late = int(metrics['LLC_Pref_Late'])
        hit = int(metrics['LLC_Pref_Hit'])
        others = issued - useful - useless - late - hit
        metrics['LLC_Pref_OTHERS'] = str(max(0, others))
        

        
    except Exception as e:
        print(f"读取文件出错 {filepath}: {e}")
        
    return metrics


def generate_advanced_plots(csv_path):
    # useful + late + useless + pre_hit + others =issued
    try:
        import pandas as pd
        import matplotlib.pyplot as plt
        import seaborn as sns
        import numpy as np
        from matplotlib.patches import Patch
    except ImportError:
        print("\n[Warning] 缺少 pandas/matplotlib/seaborn，无法绘图。")
        return

    print(f"\n[Plot] 正在基于 {csv_path} 生成分析图表...")
    df = pd.read_csv(csv_path)

    num_cols = ['IPC', 'LLC_Pref_Issued', 'LLC_Pref_Useful', 'LLC_Pref_Late', 'LLC_Pref_Useless',      'LLC_Pref_Hit', 'LLC_Pref_OTHERS',          'Size', 'Freq']
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)

    # 学术论文风格 5-class 配色
    color_useful  = '#4daf4a'   # 绿色
    color_late    = '#ffffff'   # 白色
    color_useless = '#e41a1c'   # 红色  
    color_hit     = '#bdbdbd'   # 中浅灰
    color_others  = '#e0e0e0'   # 更浅灰

    # =========================================================
    # 图 1：网格大图 - 各个 Trace 在 Size=2048 下，三类预取数量随 Freq 的柱状图变化
    # =========================================================
    target_size_for_trend = 2048
    df_trend = df[(df['Size'] == target_size_for_trend) & (df['Model'].isin(['pathfinder', 'pythia']))].copy()
    
    if not df_trend.empty:
        benchmarks = sorted(df_trend['Benchmark'].unique())
        freqs = sorted(df_trend['Freq'].unique())
        models_to_plot = ['pathfinder', 'pythia']
        
        # 自动计算子图的行列数（每行 4 个）
        cols = 4
        rows = math.ceil(len(benchmarks) / cols)
        
        fig1, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows), sharex=True)
        # 兼容单独一行或多行返回的 axes 维度问题
        if rows == 1: axes = np.array([axes]) 
        axes = axes.flatten()

        x = np.arange(len(freqs))
        width = 0.35  

        for idx, bench in enumerate(benchmarks):
            ax = axes[idx]
            b_data = df_trend[df_trend['Benchmark'] == bench]
            
            for i, model in enumerate(models_to_plot):
                m_data = b_data[b_data['Model'] == model].set_index('Freq').reindex(freqs).fillna(0)
                
                useful = m_data['LLC_Pref_Useful'].values
                late = m_data['LLC_Pref_Late'].values
                useless = m_data['LLC_Pref_Useless'].values
                hit = m_data['LLC_Pref_Hit'].values
                others = m_data['LLC_Pref_OTHERS'].values
                
                # 并排偏移量：Pathfinder 在左边，Pythia 在右边
                offset = (i - 0.5) * width
                
                ax.bar(x + offset, useful, width, color=color_useful, edgecolor='black')
                ax.bar(x + offset, late, width, bottom=useful, color=color_late, edgecolor='black', hatch='//')
                ax.bar(x + offset, useless, width, bottom=useful+late, color=color_useless, edgecolor='black', hatch='//')
                ax.bar(x + offset, hit, width, bottom=useful+late+useless, color=color_hit, edgecolor='black', hatch='//')
                ax.bar(x + offset, others, width, bottom=useful+late+useless+hit, color=color_others, edgecolor='black')

            ax.set_title(bench, fontsize=12)
            ax.set_xticks(x)
            ax.set_xticklabels(freqs, rotation=45)
            ax.grid(axis='y', linestyle='--', alpha=0.7)
            if idx % cols == 0:
                ax.set_ylabel('Absolute Prefetch Count')

        # 隐藏多余的空白子图
        for idx in range(len(benchmarks), len(axes)):
            axes[idx].set_visible(False)

        # 整体标题和图例
        fig1.suptitle(f'Dynamic Prefetch Adjustment by Freq @ Size={target_size_for_trend}\n(Left Bar: Pathfinder | Right Bar: Pythia)', 
                      fontsize=18, fontweight='bold', y=0.96)
        
        legend_elements = [
            Patch(facecolor=color_useful, edgecolor='black', label='Useful (prefetch used)'),
            Patch(facecolor=color_late, edgecolor='black', hatch='//', label='Late (prefetch untimely)'),
            Patch(facecolor=color_useless, edgecolor='black', hatch='//', label='Useless (prefetch not used)'),
            Patch(facecolor=color_hit, edgecolor='black', hatch='//', label='hit (prefetch hit by LLC)'),
            Patch(facecolor=color_others, edgecolor='black', label='others (others)')

        ]
        fig1.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 0.91), ncol=3, frameon=False, fontsize=12)
        
        # 留出顶部空间防止标题重叠
        plt.tight_layout(rect=[0, 0, 1, 0.88])
        fig1_name = f"plot_prefetch_dynamic_bars_{target_size_for_trend}.png"
        fig1.savefig(fig1_name, dpi=300)
        plt.close(fig1)
        print(f"  --> 已生成 图1 (资源感知网格柱状图): {fig1_name}")


    # =========================================================
    # 图 2：绝对预取类型的堆叠对比 与 加速比(Speedup) 对比图
    # =========================================================
    target_size, target_freq = 2048, 4800
    df_filtered = df[(df['Size'] == target_size) & (df['Freq'] == target_freq)].copy()
    
    if df_filtered.empty:
        print(f"[Plot] 错误：未找到 Size={target_size} & Freq={target_freq} 的记录。")
        return

    df_filtered = df_filtered.sort_values('Benchmark')
    benchmarks = df_filtered['Benchmark'].unique()
    pf_models = [m for m in df_filtered['Model'].unique() if m != 'no']

    if not pf_models:
        return

    # 提取 no (baseline) 的 IPC，用于计算加速比
    ipc_no = df_filtered[df_filtered['Model'] == 'no'].set_index('Benchmark')['IPC']

    fig2, ax_bar = plt.subplots(figsize=(14, 7))
    ax_line = ax_bar.twinx()  

    x = np.arange(len(benchmarks))
    width = 0.35  

    # 1. 绘制绝对值的堆叠柱状图
    for i, model in enumerate(pf_models):
        offset = (i - len(pf_models)/2 + 0.5) * width
        model_data = df_filtered[df_filtered['Model'] == model].set_index('Benchmark').reindex(benchmarks).fillna(0)

        
        useful = model_data['LLC_Pref_Useful'].values
        late = model_data['LLC_Pref_Late'].values
        useless = model_data['LLC_Pref_Useless'].values
        hit = model_data['LLC_Pref_Hit'].values
        others = model_data['LLC_Pref_OTHERS'].values
        total = model_data['LLC_Pref_Issued'].values
        ax_bar.bar(x + offset, useful, width, color=color_useful, edgecolor='black')
        ax_bar.bar(x + offset, late, width, bottom=useful, color=color_late, edgecolor='black', hatch='//')
        ax_bar.bar(x + offset, useless, width, bottom=useful+late, color=color_useless, edgecolor='black', hatch='//')
        ax_bar.bar(x + offset, hit, width, bottom=useful+late+useless, color=color_hit, edgecolor='black', hatch='//')
        ax_bar.bar(x + offset, others, width, bottom=useful+late+useless+hit, color=color_others, edgecolor='black')

        # 在柱子上方标注模型简称
        for pos, tot in zip(x + offset, total):
            if tot > 0:
                ax_bar.text(pos, - (df_filtered['LLC_Pref_Issued'].max() * 0.02), model[:4].capitalize(), 
                            ha='center', va='top', fontsize=9, color='grey')

    # 2. 绘制加速比 (Speedup) 折线图 (不再画 no)
    markers = {'pathfinder': 's', 'pythia': '^'}
    colors = {'pathfinder': '#fc8d62', 'pythia': '#8da0cb'}
    max_speedup = 1.0
    
    for model in pf_models:
        model_data = df_filtered[df_filtered['Model'] == model].set_index('Benchmark').reindex(benchmarks)
        
        # 计算相对 baseline ('no') 的加速比
        speedup = model_data['IPC'].values / ipc_no.reindex(benchmarks).values
        max_speedup = max(max_speedup, np.nanmax(speedup))
        
        ax_line.plot(x, speedup, marker=markers.get(model, 'o'), 
                     color=colors.get(model, 'black'), linewidth=2, markersize=8, 
                     alpha=0.9, label=f'{model.capitalize()} Speedup')

    # 画一条 Y=1.0 的基准参考线
    ax_line.axhline(1.0, color='grey', linestyle='--', alpha=0.6, label='Baseline (No PF)')

    # 图表装饰与轴边界调整
    ax_bar.set_ylabel('Absolute Prefetch Count (Issued)', fontsize=12, fontweight='bold')
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(benchmarks, rotation=45, ha='right', fontsize=11)
    ax_bar.set_ylim(0, df_filtered['LLC_Pref_Issued'].max() * 1.3)
    
    ax_line.set_ylabel('Speedup over Baseline', fontsize=12, fontweight='bold')
    # 动态调整折线图上下界，包含基准线 1.0
    min_y = min(0.9, np.nanmin([1.0, df_filtered[df_filtered['Model'].isin(pf_models)]['IPC'].min() / ipc_no.min()]))
    ax_line.set_ylim(min_y * 0.95, max_speedup * 1.15) 
    
    plt.title(f'Absolute Prefetch Breakdown vs Speedup @ Size={target_size}, Freq={target_freq}MHz', fontsize=16, fontweight='bold', pad=30)
    
    handles_line, labels_line = ax_line.get_legend_handles_labels()
    
    # 将两种图例合并，放置于图表正上方
    ax_bar.legend(legend_elements + handles_line, 
                  ['Useful', 'Late', 'Useless', 'hit', 'others'] + labels_line, 
                  loc='upper center', bbox_to_anchor=(0.5, 1.12), ncol=6, frameon=False, fontsize=11)
    
    # 彻底解决标题/图例重叠问题
    plt.tight_layout(rect=[0, 0, 1, 0.92]) 
    fig2_name = f"plot_absolute_breakdown_speedup_{target_size}_{target_freq}.png"
    fig2.savefig(fig2_name, dpi=300)
    plt.close(fig2)
    print(f"  --> 已生成 图2 (绝对数量拆解与加速比): {fig2_name}")


def main():
    parser = argparse.ArgumentParser(description="解析 ChampSim 日志，生成 CSV 汇总，并绘制高级对比图")
    parser.add_argument("-d", "--dir", type=str, default=DEFAULT_DIR, help=f"指定包含日志的文件夹路径 (默认: {DEFAULT_DIR})")
    parser.add_argument("-o", "--output", type=str, default="summary_results.csv", help="输出的 CSV 文件名")
    args = parser.parse_args()

    log_directory = args.dir
    output_csv = args.output

    if not os.path.isdir(log_directory):
        print(f"❌ 错误: 找不到文件夹 '{log_directory}'")
        return

    file_patterns = [os.path.join(log_directory, "*.txt"), os.path.join(log_directory, "*.log")]
    log_files = []
    for pattern in file_patterns:
        log_files.extend(glob.glob(pattern))

    if not log_files:
        print(f"⚠️ 警告: 文件夹内无日志文件。")
        return

    print(f"🔍 正在扫描文件夹并解析...")
    all_metrics = [parse_champsim_log(f) for f in log_files]

    if all_metrics:
        keys = all_metrics[0].keys()
        with open(output_csv, 'w', newline='', encoding='utf-8') as output_file:
            dict_writer = csv.DictWriter(output_file, fieldnames=keys)
            dict_writer.writeheader()
            dict_writer.writerows(all_metrics)
        print(f"✅ 解析完成，触发绘图...")
        generate_advanced_plots(output_csv)

if __name__ == "__main__":
    main()