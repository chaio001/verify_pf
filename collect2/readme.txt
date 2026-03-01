预取器性能与收敛性分析结果 (ChampSim)
本目录包含了用于解析 ChampSim 仿真结果（日志与 Trace CSV）的自动化脚本，以及生成的指标统计表和可视化图表。主要对比了不同硬件配置（LLC Size、DRAM 频率）下，Pathfinder 与 Pythia 预取器的表现差异。

1. 核心脚本与配置 (Scripts & Config)
collect.json：脚本驱动配置文件。定义了输入目录路径、匹配规则（正则表达式），将不同后缀的文件精准映射到对应的 Benchmark 和 Model (no/pathfinder/pythia)。

collect2.py：宏观指标解析脚本。负责扫描仿真输出的 .txt 日志文件，提取 IPC 以及各项预取统计信息（Requested, Useful, Useless, Late, Hit 等），并生成二维对比表格与可视化柱状图。

collect_csv.py：微观行为分析脚本。基于多进程加速，扫描高达几百万行的 .csv 行为轨迹文件。专门用于追踪模型在遇到相同特征（Delta1, Delta2, Delta3）时，其输出动作（Action/RLAction）的切换频率。

2. 宏观性能与预取效率评估 (基于 TXT 日志)
这些文件由 collect2.py 生成，所有二维表格中相同配置下的最优值已用 >> 自动高亮标出。

metric_ipc.txt：累计 IPC 对比。直观展示不同预取器对系统整体性能的加速效果。

metric_accuracy.txt：预取准确率 (Prefetch Accuracy)。计算方式为 Useful / Issued，评估预取器发出的请求中有多大比例被实际使用。

metric_coverage.txt：预取覆盖率 (Prefetch Coverage)。计算方式为 Useful / Load Access，评估预取器成功消除了多少原本会发生的 LLC Load Miss。

metric_bw_share.txt：带宽占用比 (Bandwidth Share)。计算方式为 Prefetch Miss / Total LLC Miss，评估预取器带来的额外内存带宽开销压力。

metric_breakdown.txt：预取请求成分拆解表。将每个 Benchmark 下的预取动作详细拆解为 Issued, Useful, Useless, Late, Hit, Others 的绝对计数值。

metric_summary.txt：全局数据汇总大表。包含上述所有指标的宽表，方便全选复制导入至 Excel 中进行二次处理。

3. 策略稳定性与收敛性分析 (基于 CSV Trace)
这些文件由 collect_csv.py 生成，排除了前 10M Cycles 的 Warmup 阶段，仅统计后续的有效访存指令。

csv_metric_changes.txt：动作切换总次数 (Total Changes)。统计在相同特征输入下，模型动作发生动荡（Thrashing）的绝对总次数。

csv_metric_ratio.txt：动作切换率 (Change Ratio)。计算方式为 Total Changes / Valid Instructions。

csv_interval_breakdown.txt：时间线收敛详情 (Interval Breakdown)。将 10M 以后的周期按每 1,000,000 (1M) 划分区间，记录每个区间内的动作切换次数。

4. 可视化图表 (Visualizations)
plot_breakdown_size[256/512/1024/2048/4096].png：

功能：不同硬件容量 (Size) 下的预取成分组装堆叠柱状图。

特点：X 轴为 DRAM 频率扩展，将 Pathfinder (左) 和 Pythia (右) 紧挨着并排展示。

视觉：使用标准论文配色与斜线网纹 (Hatch) 区分 Useful (绿色正向), Useless (红色负向), Late (白色警告) 等成分，帮助直观分析在不同带宽压力下，两款预取器的请求成分结构差异。