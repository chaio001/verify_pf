#ifndef LEARNING_ENGINE_HDC_BASIC_H
#define LEARNING_ENGINE_HDC_BASIC_H

#include <string.h>

#include <random>

#include "hdc_pf_helper.h"
#include "hyperstream/core/hypervector.hpp"
#include "learning_engine_hdc.h"
#include "prefetcher.h"
#define MAX_ACTIONS 64

/*
 * table format
 *      |action 0| action 1| action 2|...| action n
state 0 |
state 1 |
                |	  ____         _        _     _
                |	 / __ \       | |      | |   | |
                |	| |  | |______| |_ __ _| |__ | | ___
                |	| |  | |______| __/ _` | '_ \| |/ _ \
                |	| |__| |      | || (_| | |_) | |  __/
                |	 \___\_\       \__\__,_|_.__/|_|\___|
                |
state m |
*/

class LearningEngineHDCBasic : public LearningEngineHDC {
   private:
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

    // float** qtable;  // 二维数组
    float* hdc_qtable;

    /* tracing related knobs */
    uint32_t trace_interval;
    uint64_t trace_timestamp;
    FILE* trace;
    uint32_t action_trace_interval;
    uint64_t action_trace_timestamp;
    FILE* action_trace;
    uint64_t m_action_counter;
    uint64_t m_early_exploration_window;

    struct
    {
        struct
        {
            uint64_t called;
            uint64_t explore;
            uint64_t exploit;
            uint64_t dist[MAX_ACTIONS][2]; /* 0:explored, 1:exploited */
        } action;

        struct
        {
            uint64_t called;
            uint64_t dist[MAX_ACTIONS][2]; /* 0:explored, 1:exploited */
        } learn;

        struct
        {
            uint64_t called;
            uint64_t dist[MAX_ACTIONS][2]; /* 0:explored, 1:exploited */
        } learnTrue;

    } stats;

    // 查询 | 更新 state&action 对应的Q值
    float consultQ(uint32_t state, uint32_t action);
    void updateQ(uint32_t state, uint32_t action, float value);
    // 把某个state对应的整行Q值格式化,返回字符串
    std::string getStringQ(EncodedState state);
    // 查找某个状态下的最大动作编号
    uint32_t getMaxAction(EncodedState* encodedState);
    // 打印统计信息
    void print_aux_stats();
    // 记录某个状态的详细信息 => 写入文件
    void dump_state_trace(uint32_t state);
    // 画图
    void plot_scores();
    // 记录某个动作的详细信息 => 写入文件
    void dump_action_trace(uint32_t action);
    // 编码器
    void init_memories();
    uint32_t bucket_pc(uint64_t pc) const { return pc % HDC_N_PC; }
    uint32_t bucket_delta(int32_t d) const {
        return (d % HDC_N_DELTA + HDC_N_DELTA) % HDC_N_DELTA;
    }

   public:
    void encode_state(EncodedState* encodedState, uint64_t pc, const std::vector<int32_t>& delta_pattern);
    LearningEngineHDCBasic(Prefetcher* p, float alpha, float gamma, float epsilon, uint32_t actions, uint32_t states, uint64_t seed, std::string policy, std::string type, bool zero_init, uint64_t early_exploration_window);
    ~LearningEngineHDCBasic();

    uint32_t chooseAction(EncodedState* encodedState);
    void learn(uint32_t state1, uint32_t action1, int32_t reward, uint32_t state2, uint32_t action2);
    void updateModel(EncodedState* old_state, uint32_t action, int32_t reward);
    void updateModel_true(EncodedState* old_state, uint32_t action, int32_t reward);
    void dump_stats();
};

#endif /* LEARNING_ENGINE */
