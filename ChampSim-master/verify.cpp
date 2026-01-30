// g++ verify.cpp -o verify -std=c++17 -I /home/zxie/cyhcpp/Pythia/a_pythia/ChampSim-master/HyperStream/include/
#include <assert.h>
#include <stdio.h>
#include <string.h>
#include <strings.h>

#include <algorithm>
#include <bitset>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <map>
#include <random>
#include <sstream>
#include <vector>

#include "hyperstream/core/hypervector.hpp"
constexpr std::size_t HDC_DIM = 1024;  // 对应 D=2048
using BinaryHV = hyperstream::core::HyperVector<HDC_DIM, bool>;
using WeightVector = std::array<int8_t, HDC_DIM>;
// === Item Memories (随机固定的基向量) ===
std::vector<BinaryHV> im_pc_;
std::vector<BinaryHV> im_delta_;
// === Role Vectors (角色向量) ===
BinaryHV role_pc_lastdelta_;
BinaryHV role_delta_seq_;
// === Associative Memory (权重矩阵 M_a) ===
// 维度: [Actions][Dimensions]
std::vector<WeightVector> m_a_;
float init_value;  // 表格初始化数值

std::default_random_engine generator;           // 随机数引擎
std::bernoulli_distribution* explore;           // 伯努利分布 , 一般用于实现 ε-greedy
std::uniform_int_distribution<int>* actiongen;  // 探索 | 随机选取动作
std::mt19937_64 rng_;

static void random_init_hv(BinaryHV& hv, std::mt19937_64& rng) {
    auto& words = hv.Words();
    for (auto& w : words) w = rng();
    // 处理尾部掩码
    constexpr size_t bits = BinaryHV::Size();
    constexpr size_t word_bits = 64;
    size_t excess = words.size() * word_bits - bits;
    if (excess > 0) {
        uint64_t mask = (~0ULL >> excess);
        words.back() &= mask;
    }
}

int main() {
    constexpr size_t HDC_N_PC = 5;  // 仅生成少量用于验证
    constexpr size_t HDC_N_DELTA = 5;
    uint64_t m_seed = 123456;  // 固定种子
    float epsilon = 0.2;
    int m_actions = 16;
    generator.seed(m_seed);
    explore = new std::bernoulli_distribution(epsilon);
    actiongen = new std::uniform_int_distribution<int>(0, m_actions - 1);
    rng_.seed(m_seed);

    im_pc_.resize(HDC_N_PC);
    im_delta_.resize(HDC_N_DELTA);
    m_a_.resize(m_actions);  // 权重矩阵
    for (auto& hv : im_pc_) random_init_hv(hv, rng_);
    for (auto& hv : im_delta_) random_init_hv(hv, rng_);
    random_init_hv(role_pc_lastdelta_, rng_);
    random_init_hv(role_delta_seq_, rng_);
    for (auto& w_vec : m_a_) {
        w_vec.fill(0);
    }
    // --- 验证部分 ---

    std::cout << "=== Verification Results ===" << std::endl;

    // 验证 1: 检查 im_pc 和 im_delta 是否相等
    // 理论预期：应该不相等，因为它们使用了 RNG 序列的不同部分
    bool all_different = true;

    // 打印前几个字的哈希值以供目视检查
    std::cout << std::endl
              << "=== Visual Inspection (First 64 bits hex) ===" << std::endl;
    std::cout << "im_pc_[0]:           0x" << std::hex << im_pc_[0].Words()[0] << std::endl;
    std::cout << "im_pc_[1]:           0x" << std::hex << im_pc_[1].Words()[0] << std::endl;
    std::cout << "im_delta_[0]:        0x" << std::hex << im_delta_[0].Words()[0] << std::endl;
    std::cout << "role_pc_lastdelta_:  0x" << std::hex << role_pc_lastdelta_.Words()[0] << std::endl;
    std::cout << "role_delta_seq_:     0x" << std::hex << role_delta_seq_.Words()[0] << std::endl;
    return 0;
}