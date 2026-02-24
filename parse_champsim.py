import os
import re
import csv
import glob
import argparse

# ==========================================
# 在这里设置你的默认日志文件夹路径！
# (当你不输入任何参数直接运行时，会默认读取这个文件夹)
# ==========================================
# DEFAULT_DIR = r"C:\Users\26416\Desktop\logcsv"  # <-- 替换为你的真实路径
DEFAULT_DIR = r"/mnt/sda/cyhcpp/Pythia/a_pythia/ChampSim-master/results"  # <-- 替换为你的真实路径

def parse_champsim_log(filepath):
    """解析单个 ChampSim 日志文件，返回提取的指标字典"""
    metrics = {
        'Trace_Name': os.path.basename(filepath),
        'IPC': '', 'Instructions': '', 'Cycles': '',
        'L1D_Access': '', 'L1D_Hit': '', 'L1D_Miss': '',
        'L2C_Access': '', 'L2C_Hit': '', 'L2C_Miss': '',
        'LLC_Access': '', 'LLC_Hit': '', 'LLC_Miss': '',
        'LLC_Pref_Requested': '', 'LLC_Pref_Issued': '', 
        'LLC_Pref_Useful': '', 'LLC_Pref_Useless': '',
        # 'My_PF_Rate(%)': '', 'My_PF_Hits': '', 'My_PF_Misses': '',
        # 'Branch_Acc(%)': '', 'MPKI': '',
        'Avg_Congested_Cycle': ''
    }

    # 定义正则表达式
    p_ipc = re.compile(r"CPU 0 cumulative IPC:\s+([\d.]+)\s+instructions:\s+(\d+)\s+cycles:\s+(\d+)")
    p_l1d = re.compile(r"L1D TOTAL\s+ACCESS:\s+(\d+)\s+HIT:\s+(\d+)\s+MISS:\s+(\d+)")
    p_l2c = re.compile(r"L2C TOTAL\s+ACCESS:\s+(\d+)\s+HIT:\s+(\d+)\s+MISS:\s+(\d+)")
    p_llc = re.compile(r"LLC TOTAL\s+ACCESS:\s+(\d+)\s+HIT:\s+(\d+)\s+MISS:\s+(\d+)")
    p_llc_pref = re.compile(r"LLC PREFETCH\s+REQUESTED:\s+(\d+)\s+ISSUED:\s+(\d+)\s+USEFUL:\s+(\d+)\s+USELESS:\s+(\d+)")
    
    p_my_pf_rate = re.compile(r"My_PF_Lookup_Rate:\s+([\d.]+)%")
    p_my_pf_hits = re.compile(r"My_PF_Lookup_Hits:\s+(\d+)")
    p_my_pf_misses = re.compile(r"My_PF_Lookup_Misses:\s+(\d+)")
    
    p_branch = re.compile(r"CPU 0 Branch Prediction Accuracy:\s+([\d.]+)%\s+MPKI:\s+([\d.]+)")
    p_congest = re.compile(r"AVG_CONGESTED_CYCLE:\s+(\d+)")

    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                # 使用海象运算符 (Python 3.8+) 简化代码，如果没有的话可以改回原来的写法
                if m := p_ipc.search(line):
                    metrics['IPC'], metrics['Instructions'], metrics['Cycles'] = m.groups()
                elif m := p_llc.search(line):
                    metrics['LLC_Access'], metrics['LLC_Hit'], metrics['LLC_Miss'] = m.groups()
                elif m := p_llc_pref.search(line):
                    metrics['LLC_Pref_Requested'], metrics['LLC_Pref_Issued'], metrics['LLC_Pref_Useful'], metrics['LLC_Pref_Useless'] = m.groups()
                elif m := p_l2c.search(line):
                    metrics['L2C_Access'], metrics['L2C_Hit'], metrics['L2C_Miss'] = m.groups()
                elif m := p_l1d.search(line):
                    metrics['L1D_Access'], metrics['L1D_Hit'], metrics['L1D_Miss'] = m.groups()
                # elif m := p_my_pf_rate.search(line):
                #     metrics['My_PF_Rate(%)'] = m.group(1)
                # elif m := p_my_pf_hits.search(line):
                #     metrics['My_PF_Hits'] = m.group(1)
                # elif m := p_my_pf_misses.search(line):
                #     metrics['My_PF_Misses'] = m.group(1)
                # elif m := p_branch.search(line):
                #     metrics['Branch_Acc(%)'], metrics['MPKI'] = m.groups()
                elif m := p_congest.search(line):
                    metrics['Avg_Congested_Cycle'] = m.group(1)
    except Exception as e:
        print(f"读取文件出错 {filepath}: {e}")
        
    return metrics

def main():
    # 1. 设置命令行参数解析
    parser = argparse.ArgumentParser(description="一键解析 ChampSim 输出日志并生成 CSV 汇总表")
    parser.add_argument(
        "-d", "--dir", 
        type=str, 
        default=DEFAULT_DIR, 
        help=f"指定包含日志的文件夹路径 (默认: {DEFAULT_DIR})"
    )
    parser.add_argument(
        "-o", "--output", 
        type=str, 
        default="summary_results.csv", 
        help="输出的 CSV 文件名 (默认保存在当前运行目录下)"
    )
    args = parser.parse_args()

    log_directory = args.dir
    output_csv = args.output

    # 2. 检查目录是否存在
    if not os.path.isdir(log_directory):
        print(f"❌ 错误: 找不到文件夹 '{log_directory}'，请检查路径是否正确！")
        return

    # 3. 寻找日志文件
    file_patterns = [os.path.join(log_directory, "*.txt"), os.path.join(log_directory, "*.log")]
    log_files = []
    for pattern in file_patterns:
        log_files.extend(glob.glob(pattern))

    if not log_files:
        print(f"⚠️ 警告: 在文件夹 '{log_directory}' 中没有找到任何 .txt 或 .log 文件。")
        return

    print(f"🔍 正在扫描文件夹: {log_directory}")
    print(f"📂 找到 {len(log_files)} 个日志文件，开始解析...")

    # 4. 提取并保存数据
    all_metrics = [parse_champsim_log(f) for f in log_files]

    if all_metrics:
        keys = all_metrics[0].keys()
        with open(output_csv, 'w', newline='', encoding='utf-8') as output_file:
            dict_writer = csv.DictWriter(output_file, fieldnames=keys)
            dict_writer.writeheader()
            dict_writer.writerows(all_metrics)
        print(f"✅ 解析大功告成！数据已保存至: {os.path.abspath(output_csv)}")

if __name__ == "__main__":
    main()
    
# python parse_champsim.py
# python parse_champsim.py -d "C:\Users\26416\Desktop\spec-selected\run2"
# python parse_champsim.py -d "C:\logs" -o "experiment_pythia_vs_snn.csv"