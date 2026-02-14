#!/bin/bash
# ./ml_prefetch_sim.py generate ./gap_spec_traces/450.soplex-s0.txt.xz     ./pathfinder_prefetches_gap_spec/prefetch_14th_rerun_450.soplex-s0.trace.txt      --model temp       
# ./ml_prefetch_sim.py generate ./gap_spec_traces/cc-5.txt.xz              ./pathfinder_prefetches_gap_spec/prefetch_14th_rerun_cc-5.trace.txt               --model temp
# ./ml_prefetch_sim.py generate ./gap_spec_traces/471.omnetpp-s1.txt.xz    ./pathfinder_prefetches_gap_spec/prefetch_14th_rerun_471.omnetpp-s1.trace.txt     --model temp           
# ./ml_prefetch_sim.py generate ./gap_spec_traces/482.sphinx3-s0.txt.xz    ./pathfinder_prefetches_gap_spec/prefetch_14th_rerun_482.sphinx3-s0.trace.txt     --model temp           
# ./ml_prefetch_sim.py generate ./gap_spec_traces/605.mcf-s1.txt.xz        ./pathfinder_prefetches_gap_spec/prefetch_14th_rerun_605.mcf-s1.trace.txt         --model temp       
# ./ml_prefetch_sim.py generate ./gap_spec_traces/bfs-10.txt.xz            ./pathfinder_prefetches_gap_spec/prefetch_14th_rerun_bfs-10.trace.txt             --model temp   
# ./ml_prefetch_sim.py generate ./gap_spec_traces/623.xalancbmk-s1.txt.xz  ./pathfinder_prefetches_gap_spec/prefetch_14th_rerun_623.xalancbmk-s1.trace.txt   --model temp           
# ./ml_prefetch_sim.py generate ./gap_spec_traces/473.astar-s1.txt.xz      ./pathfinder_prefetches_gap_spec/prefetch_14th_rerun_473.astar-s1.trace.txt       --model temp       

./ml_prefetch_sim.py run ./gap_spec_traces/bfs-10.trace.gz --prefetch ./pathfinder_prefetches_gap_spec/prefetch_14th_rerun_bfs-10.trace.txt --num-instructions 61219343             
./ml_prefetch_sim.py run ./gap_spec_traces/cc-5.trace.gz --prefetch ./pathfinder_prefetches_gap_spec/prefetch_14th_rerun_cc-5.trace.txt --num-instructions 20700644               
./ml_prefetch_sim.py run ./gap_spec_traces/450.soplex-s0.trace.gz --prefetch ./pathfinder_prefetches_gap_spec/prefetch_14th_rerun_450.soplex-s0.trace.txt --num-instructions 29478984      
./ml_prefetch_sim.py run ./gap_spec_traces/482.sphinx3-s0.trace.gz --prefetch ./pathfinder_prefetches_gap_spec/prefetch_14th_rerun_482.sphinx3-s0.trace.txt --num-instructions 84756213     
./ml_prefetch_sim.py run ./gap_spec_traces/623.xalancbmk-s1.trace.xz --prefetch ./pathfinder_prefetches_gap_spec/prefetch_14th_rerun_623.xalancbmk-s1.trace.txt --num-instructions 52596273   
./ml_prefetch_sim.py run ./gap_spec_traces/473.astar-s1.trace.gz --prefetch ./pathfinder_prefetches_gap_spec/prefetch_14th_rerun_473.astar-s1.trace.txt --num-instructions 98033975       
./ml_prefetch_sim.py run ./gap_spec_traces/605.mcf-s1.trace.xz --prefetch ./pathfinder_prefetches_gap_spec/prefetch_14th_rerun_605.mcf-s1.trace.txt --num-instructions 38467912         
./ml_prefetch_sim.py run ./gap_spec_traces/471.omnetpp-s1.trace.gz --prefetch ./pathfinder_prefetches_gap_spec/prefetch_14th_rerun_471.omnetpp-s1.trace.txt --num-instructions 54612142     


# <./pathfinder_prefetches_gap_spec/prefetch_14th_rerun_471.omnetpp-s1.trace.txt bin/hashed_perceptron-no-no-no-from_file-lru-1core_3200_2048_debug -prefetch_warmup_instructions 10000000 -simulation_instructions 54612142 -seed 1083 -traces ./gap_spec_traces/471.omnetpp-s1.trace.gz > ./results/471.omnetpp-s1.trace.gz-from_file_3200_2048_debug.txt 2>&1
# <./pathfinder_prefetches_gap_spec/prefetch_14th_rerun_471.omnetpp-s1.trace.txt bin/hashed_perceptron-no-no-no-from_file-lru-1core_6400_2048_debug -prefetch_warmup_instructions 10000000 -simulation_instructions 54612142 -seed 1083 -traces ./gap_spec_traces/471.omnetpp-s1.trace.gz > ./results/471.omnetpp-s1.trace.gz-from_file_6400_2048_debug.txt 2>&1


# mkdir -p pathfinder_prefetches_gap_spec

# # script to run pathfinder on gap and spec
# find -L "./gap_spec_traces" -type f -name "*.txt.xz" | while read xz; do
#   file_variable="$(basename $xz .txt.xz)"
#   file_cc="cc-5"
#   file_605mcf="605.mcf-s1"
#   file_bfs10="bfs-10"
#   file_450soplex="450.soplex-s0"
#   file_623xalan="623.xalancbmk-s1"
#   file_471omnetpp="471.omnetpp-s1"
#   file_482sphinx3="482.sphinx3-s0"
#   file_473astar="473.astar-s1"


#   # ./ml_prefetch_sim.py generate "$xz" ./pathfinder_prefetches_gap_spec/prefetch_14th_rerun_"$(basename "$xz" .txt.xz)".txt --model temp

#   if [[ "$file_variable" == "$file_cc" ]]; then
#   # echo ./gap_spec_traces/"$file_variable".trace.gz
#   ./ml_prefetch_sim.py generate "$xz" ./pathfinder_prefetches_gap_spec/prefetch_14th_rerun_"$(basename "$xz" .txt.xz)".txt --model temp
#     ./ml_prefetch_sim.py run ./gap_spec_traces/"$file_variable".trace.gz --prefetch ./pathfinder_prefetches_gap_spec/prefetch_14th_rerun_"$file_variable".txt --num-instructions 20700644
#     # ./ml_prefetch_sim.py run ./gap_spec_traces/"$file_variable".trace.gz --prefetch ./pathfinder_prefetches_gap_spec/prefetch_14th_rerun_cc-5.trace.txt --num-instructions 20700644
#   fi

#   # if [[ "$file_variable" == "$file_605mcf" ]]; then
#   #   ./ml_prefetch_sim.py run ./gap_spec_traces/"$file_variable".trace.xz --prefetch ./pathfinder_prefetches_gap_spec/prefetch_14th_rerun_"$file_variable".txt --num-instructions 38467912
#   #   # ./ml_prefetch_sim.py run ./gap_spec_traces/"$file_variable".trace.xz --prefetch ./pathfinder_prefetches_gap_spec/prefetch_14th_rerun_605.mcf-s1.trace.txt --num-instructions 38467912
#   # fi

#   # if [[ "$file_variable" == "$file_bfs10" ]]; then
#   #   ./ml_prefetch_sim.py run ./gap_spec_traces/"$file_variable".trace.gz --prefetch ./pathfinder_prefetches_gap_spec/prefetch_14th_rerun_"$file_variable".txt --num-instructions 61219343
#   # fi

#   # if [[ "$file_variable" == "$file_450soplex" ]]; then
#   #   ./ml_prefetch_sim.py run ./gap_spec_traces/"$file_variable".trace.gz --prefetch ./pathfinder_prefetches_gap_spec/prefetch_14th_rerun_"$file_variable".txt --num-instructions 29478984
#   # fi

#   # if [[ "$file_variable" == "$file_623xalan" ]]; then
#   #   ./ml_prefetch_sim.py run ./gap_spec_traces/"$file_variable".trace.xz --prefetch ./pathfinder_prefetches_gap_spec/prefetch_14th_rerun_"$file_variable".txt --num-instructions 52596273
#   # fi

#   # if [[ "$file_variable" == "$file_471omnetpp" ]]; then
#   #   ./ml_prefetch_sim.py run ./gap_spec_traces/"$file_variable".trace.gz --prefetch ./pathfinder_prefetches_gap_spec/prefetch_14th_rerun_"$file_variable".txt --num-instructions 54612142
#   # fi

#   # if [[ "$file_variable" == "$file_482sphinx3" ]]; then
#   #   ./ml_prefetch_sim.py run ./gap_spec_traces/"$file_variable".trace.gz --prefetch ./pathfinder_prefetches_gap_spec/prefetch_14th_rerun_"$file_variable".txt --num-instructions 84756213
#   # fi

#   # if [[ "$file_variable" == "$file_473astar" ]]; then
#   #   ./ml_prefetch_sim.py run ./gap_spec_traces/"$file_variable".trace.gz --prefetch ./pathfinder_prefetches_gap_spec/prefetch_14th_rerun_"$file_variable".txt --num-instructions 98033975
#   # fi

# done

