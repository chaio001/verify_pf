cd /mnt/sda/cyhcpp/Pythia/a_pythia
############################################################################################################################################################
sed -i 's/#define LLC_SET NUM_CPUS\*2048/#define LLC_SET NUM_CPUS*2048/' /mnt/sda/cyhcpp/Pythia/a_pythia/ChampSim-master/inc/cache.h
conda activate pythia_cpp
cd /mnt/sda/cyhcpp/Pythia/a_pythia/ChampSim-master
./ml_prefetch_sim.py build
cd /mnt/sda/cyhcpp/Pythia/a_pythia
cp ChampSim-master/bin/hashed_perceptron-no-no-no-no-lru-1core ChampSim-master/bin/hashed_perceptron-no-no-no-no-lru-1core_2048
cp ChampSim-master/bin/hashed_perceptron-no-no-no-from_file-lru-1core ChampSim-master/bin/hashed_perceptron-no-no-no-from_file-lru-1core_2048
cp ChampSim-master/bin/hashed_perceptron-no-no-no-pathfinder-lru-1core ChampSim-master/bin/hashed_perceptron-no-no-no-pathfinder-lru-1core_2048
sed -i 's/#define LLC_SET NUM_CPUS\*2048/#define LLC_SET NUM_CPUS*2048/' /mnt/sda/cyhcpp/Pythia/a_pythia/ChampSim-master/inc/cache.h
############################################################################################################################################################
sed -i 's/#define LLC_SET NUM_CPUS\*2048/#define LLC_SET NUM_CPUS*1024/' /mnt/sda/cyhcpp/Pythia/a_pythia/ChampSim-master/inc/cache.h
conda activate pythia_cpp
cd /mnt/sda/cyhcpp/Pythia/a_pythia/ChampSim-master
./ml_prefetch_sim.py build
cd /mnt/sda/cyhcpp/Pythia/a_pythia
cp ChampSim-master/bin/hashed_perceptron-no-no-no-no-lru-1core ChampSim-master/bin/hashed_perceptron-no-no-no-no-lru-1core_1024
cp ChampSim-master/bin/hashed_perceptron-no-no-no-from_file-lru-1core ChampSim-master/bin/hashed_perceptron-no-no-no-from_file-lru-1core_1024
cp ChampSim-master/bin/hashed_perceptron-no-no-no-pathfinder-lru-1core ChampSim-master/bin/hashed_perceptron-no-no-no-pathfinder-lru-1core_1024
sed -i 's/#define LLC_SET NUM_CPUS\*1024/#define LLC_SET NUM_CPUS*2048/' /mnt/sda/cyhcpp/Pythia/a_pythia/ChampSim-master/inc/cache.h
############################################################################################################################################################
sed -i 's/#define LLC_SET NUM_CPUS\*2048/#define LLC_SET NUM_CPUS*512/' /mnt/sda/cyhcpp/Pythia/a_pythia/ChampSim-master/inc/cache.h
conda activate pythia_cpp
cd /mnt/sda/cyhcpp/Pythia/a_pythia/ChampSim-master
./ml_prefetch_sim.py build
cd /mnt/sda/cyhcpp/Pythia/a_pythia
cp ChampSim-master/bin/hashed_perceptron-no-no-no-no-lru-1core ChampSim-master/bin/hashed_perceptron-no-no-no-no-lru-1core_512
cp ChampSim-master/bin/hashed_perceptron-no-no-no-from_file-lru-1core ChampSim-master/bin/hashed_perceptron-no-no-no-from_file-lru-1core_512
cp ChampSim-master/bin/hashed_perceptron-no-no-no-pathfinder-lru-1core ChampSim-master/bin/hashed_perceptron-no-no-no-pathfinder-lru-1core_512
sed -i 's/#define LLC_SET NUM_CPUS\*512/#define LLC_SET NUM_CPUS*2048/' /mnt/sda/cyhcpp/Pythia/a_pythia/ChampSim-master/inc/cache.h
############################################################################################################################################################
sed -i 's/#define LLC_SET NUM_CPUS\*2048/#define LLC_SET NUM_CPUS*256/' /mnt/sda/cyhcpp/Pythia/a_pythia/ChampSim-master/inc/cache.h
conda activate pythia_cpp
cd /mnt/sda/cyhcpp/Pythia/a_pythia/ChampSim-master
./ml_prefetch_sim.py build
cd /mnt/sda/cyhcpp/Pythia/a_pythia
cp ChampSim-master/bin/hashed_perceptron-no-no-no-no-lru-1core ChampSim-master/bin/hashed_perceptron-no-no-no-no-lru-1core_256
cp ChampSim-master/bin/hashed_perceptron-no-no-no-from_file-lru-1core ChampSim-master/bin/hashed_perceptron-no-no-no-from_file-lru-1core_256
cp ChampSim-master/bin/hashed_perceptron-no-no-no-pathfinder-lru-1core ChampSim-master/bin/hashed_perceptron-no-no-no-pathfinder-lru-1core_256
sed -i 's/#define LLC_SET NUM_CPUS\*256/#define LLC_SET NUM_CPUS*2048/' /mnt/sda/cyhcpp/Pythia/a_pythia/ChampSim-master/inc/cache.h
############################################################################################################################################################
sed -i 's/#define LLC_SET NUM_CPUS\*2048/#define LLC_SET NUM_CPUS*4096/' /mnt/sda/cyhcpp/Pythia/a_pythia/ChampSim-master/inc/cache.h
conda activate pythia_cpp
cd /mnt/sda/cyhcpp/Pythia/a_pythia/ChampSim-master
./ml_prefetch_sim.py build
cd /mnt/sda/cyhcpp/Pythia/a_pythia
cp ChampSim-master/bin/hashed_perceptron-no-no-no-no-lru-1core ChampSim-master/bin/hashed_perceptron-no-no-no-no-lru-1core_4096
cp ChampSim-master/bin/hashed_perceptron-no-no-no-from_file-lru-1core ChampSim-master/bin/hashed_perceptron-no-no-no-from_file-lru-1core_4096
cp ChampSim-master/bin/hashed_perceptron-no-no-no-pathfinder-lru-1core ChampSim-master/bin/hashed_perceptron-no-no-no-pathfinder-lru-1core_4096
sed -i 's/#define LLC_SET NUM_CPUS\*4096/#define LLC_SET NUM_CPUS*2048/' /mnt/sda/cyhcpp/Pythia/a_pythia/ChampSim-master/inc/cache.h
############################################################################################################################################################







