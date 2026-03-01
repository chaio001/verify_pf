#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import re
import sys

try:
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
except ImportError:
    sys.exit("[Error] 请先安装绘图依赖: pip install pandas seaborn matplotlib")

def parse_txt_to_df(file_path: str) -> pd.DataFrame:
    """
    解析人类可读的 ASCII 文本表格，将其转换为结构化的 DataFrame
    """
    print(f"[info] 正在解析数据文件: {file_path} ...")
    data = []
    
    # 正则表达式提取规则
    freq_re = re.compile(r"========== DRAM Frequency:\s+(\d+)\s+MHz ==========")
    size_re = re.compile(r"\|---\s+Size:\s+(\d+)\s+---")
    
    current_freq = None
    col_mappings = []  # 记录每一列对应的是哪个 Size 和哪个 Model

    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue

        # 1. 寻找频率段落起始标记
        m_freq = freq_re.search(line)
        if m_freq:
            current_freq = int(m_freq.group(1))
            i += 1
            
            # 2. 读取下一行的 Size
            line_size = lines[i]
            sizes = [int(s) for s in size_re.findall(line_size)]
            i += 1
            
            # 3. 读取下一行的 Models
            line_models = lines[i]
            # 把竖线替换为空格，切分提取出所有模型名
            tokens = line_models.replace('|', ' ').split()
            # 过滤掉表头的 'Benchmark'
            models = tokens[1:] if tokens and tokens[0] == 'Benchmark' else tokens
            
            # 构建列映射 (将铺平的每一列映射回对应的 Size 和 Model)
            if sizes and models:
                num_models_per_size = len(models) // len(sizes)
                col_mappings = []
                for idx, m in enumerate(models):
                    sz = sizes[idx // num_models_per_size]
                    col_mappings.append((sz, m))
            
            i += 1
            # 4. 跳过虚线分隔符 '----'
            if i < len(lines) and lines[i].startswith('----'):
                i += 1
            continue
        
        # 5. 如果当前已经在一个频率段落内，并且不是新段落的开头，则解析具体数据行
        if current_freq is not None and not line.startswith('===='):
            tokens = line.replace('|', ' ').split()
            if tokens:
                benchmark = tokens[0]
                values = tokens[1:]
                
                # 遍历后面的数值列
                for col_idx, val_str in enumerate(values):
                    if col_idx >= len(col_mappings):
                        break
                    # 跳过没有数据 '-' 的情况
                    if val_str == '-':
                        continue
                    
                    # 剥离表示 local_max 的 '>>'
                    clean_val = val_str.replace('>>', '')
                    try:
                        ipc = float(clean_val)
                        sz, mod = col_mappings[col_idx]
                        data.append({
                            "Benchmark": benchmark,
                            "Freq": current_freq,
                            "Size": sz,
                            "Model": mod,
                            "IPC": ipc
                        })
                    except ValueError:
                        pass
        i += 1

    df = pd.DataFrame(data)
    print(f"[info] 解析完毕，共提取 {len(df)} 条有效数据记录。")
    return df

def generate_visualizations(df: pd.DataFrame):
    """
    基于 DataFrame 生成高清分析图表
    """
    if df.empty:
        print("[warning] 没有足够的数据用于生成图表。")
        return

    print("[info] 正在生成可视化图表...")
    
    # ---------------------------------------------------------
    # 图 1：选择最高频率，绘制 IPC - Size 扩展趋势折线图
    # ---------------------------------------------------------
    # max_freq = df['Freq'].max()
    max_freq = 3200
    df_max_freq = df[df['Freq'] == max_freq]
    
    if not df_max_freq.empty:
        sns.set_theme(style="whitegrid")
        g = sns.relplot(
            data=df_max_freq, x="Size", y="IPC", hue="Model", 
            style="Model", markers=True, dashes=False,
            col="Benchmark", col_wrap=4, kind="line", 
            height=3, aspect=1.2, palette="Set1"
        )
        g.fig.suptitle(f"IPC Scaling vs Size @ {max_freq} MHz", y=1.05, fontsize=16, fontweight='bold')
        
        for ax in g.axes.flat:
            ax.set_xscale('log', base=2)
            sizes = sorted(df_max_freq['Size'].unique())
            ax.set_xticks(sizes)
            ax.set_xticklabels(sizes)
            
        plt.savefig(f"plot_ipc_scaling_{max_freq}MHz.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  --> 已生成折线图: plot_ipc_scaling_{max_freq}MHz.png")

    # ---------------------------------------------------------
    # 图 2：固定 Size=2048，绘制 IPC - Freq 扩展趋势折线图
    # ---------------------------------------------------------
    df_2048 = df[df['Size'] == 2048]
    
    if not df_2048.empty:
        sns.set_theme(style="whitegrid")
        g2 = sns.relplot(
            data=df_2048, x="Freq", y="IPC", hue="Model", 
            style="Model", markers=True, dashes=False,
            col="Benchmark", col_wrap=4, kind="line", 
            height=3, aspect=1.2, palette="Set2"
        )
        g2.fig.suptitle("IPC Scaling vs Freq @ Size 2048", y=1.05, fontsize=16, fontweight='bold')
        
        for ax in g2.axes.flat:
            ax.set_xscale('log', base=2)
            freqs = sorted(df_2048['Freq'].unique())
            ax.set_xticks(freqs)
            ax.set_xticklabels(freqs)
            ax.tick_params(axis='x', rotation=45)
            
        plt.savefig("plot_ipc_scaling_size2048.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  --> 已生成折线图: plot_ipc_scaling_size2048.png")

    # ---------------------------------------------------------
    # 图 3：Pythia 和 Pathfinder 相对 Baseline 的整体加速比热力图
    # ---------------------------------------------------------
    pivot_df = df.pivot_table(index=["Benchmark", "Freq", "Size"], columns="Model", values="IPC").reset_index()
    
    if 'no' in pivot_df.columns:
        models_to_compare = [m for m in ['pythia', 'pathfinder'] if m in pivot_df.columns]
        
        for model in models_to_compare:
            # 计算加速比百分比 ( (model/no - 1) * 100 )
            pivot_df[f'{model}_Speedup'] = (pivot_df[model] / pivot_df['no'] - 1) * 100
            
            # 计算全 Benchmark 均值
            heatmap_data = pivot_df.pivot_table(index="Freq", columns="Size", values=f"{model}_Speedup", aggfunc="mean")
            heatmap_data = heatmap_data.sort_index(ascending=False) # 频率倒序，高频在上方

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
            print(f"  --> 已生成热力图: {out_name}")

def main():
    parser = argparse.ArgumentParser(description="从 IPC 文本表格生成分析图表")
    parser.add_argument("-i", "--input", required=True, help="输入的文本表格文件路径 (例如: ipc_pivot.txt)")
    args = parser.parse_args()

    # 1. 解析文本为 DataFrame
    df = parse_txt_to_df(args.input)
    
    # 2. 生成图表
    generate_visualizations(df)
    print("\n[info] 所有图表生成完毕！")

if __name__ == "__main__":
    main()

# python plot_from_txt.py -i ipc_pivot_pythia_adjust.txt