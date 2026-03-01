import os

# 1. 定义 Trace 文件及其对应的指令数 (num-instructions)
# 格式: (文件名, 指令数, 简写名称用于拼接pathfinder路径)
traces = [
    ("bfs-10.trace.gz", 61219343, "bfs-10"),
    ("cc-5.trace.gz", 20700644, "cc-5"),
    ("450.soplex-s0.trace.gz", 29478984, "450.soplex-s0"),
    ("482.sphinx3-s0.trace.gz", 84756213, "482.sphinx3-s0"),
    ("623.xalancbmk-s1.trace.xz", 52596273, "623.xalancbmk-s1"),
    ("473.astar-s1.trace.gz", 98033975, "473.astar-s1"),
    ("605.mcf-s1.trace.xz", 38467912, "605.mcf-s1"),
    ("471.omnetpp-s1.trace.gz", 54612142, "471.omnetpp-s1"),
]

# 2. 定义 Only 选项列表

# only_options = [
#     'no256','pathfinder256','pythia256'
#     ,'no512','pathfinder512','pythia512'
#     ,'no1024','pathfinder1024','pythia1024'
#     ,'no2048','pathfinder2048','pythia2048'
#     ,'no4096','pathfinder4096','pythia4096'
#     # ,'no8192','pathfinder8192','pythia8192'
# ]

only_options = [
    'pythia256'
    ,'pythia512'
    ,'pythia1024'
    ,'pythia2048'
    ,'pythia4096'
    # ,'pythia8192'
]

# only_options = [
#     'pathfinder512'
# ]
# only_options = [
#     'pathfinder256',
#     'pathfinder512',
#     'pathfinder1024',
#     'pathfinder2048',
#     'pathfinder4096',
#     'pathfinder8192'
#     ]
# only_options = [
#     'no256','pathfinder256','pythia256',
#     'no1024','pathfinder1024','pythia1024',
#     'no2048','pathfinder2048','pythia2048'
# ]
# 3. 定义频率列表
# dram_freqs = [100, 200, 400, 800, 1600, 3200, 6400, 12800]
# dram_freqs = [ 100, 200, 400, 800, 1600, 3200, 6400, 12800, 25600]
dram_freqs = [ 150, 300, 600, 1200, 2400, 4800, 9600]
# dram_freqs = [ 3200]
# dram_freqs = [ 1200, 3200, 12800]
# dram_freqs = [ 200,  800, 1600,  3200,  12800]

output_file = "ChampSim-master/run_all.sh"

with open(output_file, "w") as f:
    f.write("#!/bin/bash\n\n")
    
    for trace_file, num_ins, short_name in traces:
        for only_val in only_options:
            # --- 核心逻辑：根据 only 类型确定 prefetch 参数 ---
            if only_val.startswith("pathfinder"):
                # 如果是 pathfinder，拼接特定路径
                prefetch_param = f"../Pathfinder/ChampSim/pathfinder_prefetches_gap_spec/prefetch_14th_rerun_{short_name}.trace.txt"
            else:
                # no 或 pythia 使用默认值
                prefetch_param = "test_trace"
            # ----------------------------------------------

            for freq in dram_freqs:
                cmd = (
                    f"./ml_prefetch_sim.py run ./gap_spec_traces/{trace_file} "
                    f"--prefetch {prefetch_param} "
                    f"--num-instructions {num_ins} "
                    f"--only {only_val} "
                    f"--dram-io-freq {freq}"
                )
                f.write(cmd + "\n")


print(f"生成文件: {output_file}")