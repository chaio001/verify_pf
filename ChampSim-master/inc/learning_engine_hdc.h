#ifndef LEARNING_ENGINE_HDC_H
#define LEARNING_ENGINE_HDC_H
// 定义关于HDC-预取的公用的数据与方法
#include <cstdint>
#include <map>
#include <random>
#include <vector>

#include "hyperstream/core/hypervector.hpp"
#include "prefetcher.h"

namespace knob {
extern std::size_t hdcpf_hdc_dim;
extern std::size_t hdcpf_hdc_n_pc;
extern std::size_t hdcpf_hdc_n_delta;
extern int hdcpf_hdc_max_weight;
extern int hdcpf_hdc_min_weight;
extern float hdcpf_threshold;
extern float hdcpf_update_threshold;
extern bool le_enable_trace;
extern uint32_t le_trace_interval;
extern std::string le_trace_file_name;
extern uint32_t le_trace_state;
extern bool le_enable_score_plot;
extern std::vector<int32_t> le_plot_actions;
extern std::string le_plot_file_name;
extern bool le_enable_action_trace;
extern uint32_t le_action_trace_interval;
extern std::string le_action_trace_name;
extern bool le_enable_action_plot;
}  // namespace knob

// === 超参数定义 ===
constexpr std::size_t HDC_DIM = 1024;  // 对应 D=2048
// constexpr std::size_t HDC_DIM = 4096;  // 对应 D=2048
// constexpr std::size_t HDC_N_PC = 1024;    // 对应 N_PC
// constexpr std::size_t HDC_N_DELTA = 128;  // 对应 N_DELTA
// constexpr int HDC_MAX_WEIGHT = 7;         // Clamp Max
// constexpr int HDC_MIN_WEIGHT = -8;        // Clamp Min

// const std::size_t HDC_DIM = knob::hdcpf_hdc_dim;          // 对应 D=2048
const std::size_t HDC_N_PC = knob::hdcpf_hdc_n_pc;        // 对应 N_PC
const std::size_t HDC_N_DELTA = knob::hdcpf_hdc_n_delta;  // 对应 N_DELTA
const int HDC_MAX_WEIGHT = knob::hdcpf_hdc_max_weight;    // Clamp Max
const int HDC_MIN_WEIGHT = knob::hdcpf_hdc_min_weight;    // Clamp Min

// === 类型定义 ===
// 输入状态使用高效的 Binary HyperVector
using BinaryHV = hyperstream::core::HyperVector<HDC_DIM, bool>;

// 权重矩阵使用 int8_t (为了支持累加和 Clamp)
// 我们手动管理这个，因为库里没有现成的 int8 向量类
using WeightVector = std::array<int8_t, HDC_DIM>;

#define MAX_ACTIONS 64

enum Policy {
    InvalidPolicy = 0,
    EGreedy,

    NumPolicies
};

enum LearningType {
    InvalidLearningType = 0,
    QLearning,
    SARSA,

    NumLearningTypes
};

const char* MapPolicyString(Policy policy);
const char* MapLearningTypeString(LearningType type);

class LearningEngineHDC {
   protected:
    Prefetcher* m_parent;  // 预取器父类prefetch
    float m_alpha;         // 学习率
    float m_gamma;         // 折扣因子
    float m_epsilon;       // 探索度
    uint32_t m_actions;    // 预取动作数量
    uint32_t m_states;     // 状态数量
    uint64_t m_seed;       // 随机种子
    Policy m_policy;       // 探索策略
    LearningType m_type;   // 学习策略

    // constexpr std::size_t HDC_DIM = 2048;     // 对应 D=2048
    std::size_t HDC_N_PC = 1024;    // 对应 N_PC
    std::size_t HDC_N_DELTA = 128;  // 对应 N_DELTA
    int HDC_MAX_WEIGHT = 7;         // Clamp Max
    int HDC_MIN_WEIGHT = -8;        // Clamp Min

   protected:
    LearningType parseLearningType(std::string str);
    Policy parsePolicy(std::string str);

   public:
    LearningEngineHDC(Prefetcher* p, float alpha, float gamma, float epsilon, uint32_t actions, uint32_t states, uint64_t seed, std::string policy, std::string type);
    virtual ~LearningEngineHDC() {};
    virtual void dump_stats() = 0;

    inline void setAlpha(float alpha) { m_alpha = alpha; }
    inline float getAlpha() { return m_alpha; }
    inline void setGamma(float gamma) { m_gamma = gamma; }
    inline float getGamma() { return m_gamma; }
    inline void setEpsilon(float epsilon) { m_epsilon = epsilon; }
    inline float getEpsilon() { return m_epsilon; }
    inline void setStates(uint32_t states) { m_states = states; }
    inline uint32_t getStates() { return m_states; }
    inline void setActions(uint32_t actions) { m_actions = actions; }
    inline uint32_t getActions() { return m_actions; }
};

#endif  // LEARNING_ENGINE_HDC_H