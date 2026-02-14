echo 1 ; sleep 5;
echo 1 ; sleep 5;
echo 1 ; sleep 5;
echo 1 ; sleep 5;
echo 1 ; sleep 5;
echo 1 ; sleep 5;
echo 1 ; sleep 5;
echo 1 ; sleep 5;

#parallel -j 16 --joblog job.log <./trace.sh
#parallel -j 16 --ungroup --joblog job.log <./run_pathfinder_gap_spec.sh
#parallel -j 16 --ungroup  < ./trace.sh
