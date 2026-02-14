#include "learning_engine_hdc_basic.h"

#include <assert.h>
#include <stdio.h>
#include <string.h>
#include <strings.h>

#include <sstream>

#include "hdc_pf.h"
// #include "velma.h"
#include "hdc_util.h"
#include "hyperstream/core/ops.hpp"

using namespace hyperstream::core;
#if 0
#define LOCKED(...)     \
    {                   \
        fflush(stdout); \
        __VA_ARGS__;    \
        fflush(stdout); \
    }
#define LOGID() fprintf(stdout, "[%25s@%3u] ", \
                        __FUNCTION__, __LINE__);
#define MYLOG(...) LOCKED(LOGID(); fprintf(stdout, __VA_ARGS__); fprintf(stdout, "\n");)
#else
#define MYLOG(...) \
    {              \
    }
#endif

// constexpr std::size_t HDC_DIM = 2048;     // 对应 D=2048
// constexpr std::size_t HDC_N_PC = 1024;    // 对应 N_PC
// constexpr std::size_t HDC_N_DELTA = 128;  // 对应 N_DELTA
// constexpr int HDC_MAX_WEIGHT = 7;         // Clamp Max
// constexpr int HDC_MIN_WEIGHT = -8;        // Clamp Min

// std::vector<int32_t> Actions;
// std::unordered_map<int32_t, int32_t> action_to_index_map;
namespace knob {
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
extern uint32_t hdcpf_max_actions;
extern vector<int32_t> hdcpf_actions;
}  // namespace knob

// 辅助函数：随机填充 BinaryHV
// 注意：Python TorchHD 的 bipolar 随机是 {-1, 1}
// 在 Binary 中，0 对应 +1, 1 对应 -1 (异或逻辑)

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

LearningEngineHDCBasic::LearningEngineHDCBasic(Prefetcher* parent, float alpha, float gamma, float epsilon, uint32_t actions, uint32_t states, uint64_t seed, std::string policy, std::string type, bool zero_init, uint64_t early_exploration_window)
    : LearningEngineHDC(parent, alpha, gamma, epsilon, actions, states, seed, policy, type) {
    // Actions.resize(knob::hdcpf_max_actions, 0);
    // std::copy(knob::hdcpf_actions.begin(), knob::hdcpf_actions.end(), Actions.begin());
    // // --- 新增代码 Start ---
    // // 构建反向查找表：Key = Delta值, Value = Actions中的下标
    // action_to_index_map.clear();
    // for (size_t i = 0; i < Actions.size(); ++i) {
    //     action_to_index_map[Actions[i]] = (int32_t)i;
    // }
    // // --- 新增代码 End ---

    hdc_qtable = (float*)calloc(m_actions, sizeof(float));
    assert(hdc_qtable);
    /* init Q-table */
    if (zero_init) {
        init_value = 0;
    } else {
        init_value = (float)1ul / (1 - gamma);
    }
    for (uint32_t i = 0; i < m_actions; ++i) {
        hdc_qtable[i] = init_value;
    }

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

    if (knob::le_enable_trace) {
        trace_interval = 0;
        trace_timestamp = 0;
        trace = fopen(knob::le_trace_file_name.c_str(), "w");
        assert(trace);
    }

    if (knob::le_enable_action_trace) {
        action_trace_interval = 0;
        action_trace_timestamp = 0;
        action_trace = fopen(knob::le_action_trace_name.c_str(), "w");
        assert(action_trace);
    }

    m_early_exploration_window = early_exploration_window;
    m_action_counter = 0;

    bzero(&stats, sizeof(stats));
}

LearningEngineHDCBasic::~LearningEngineHDCBasic() {
    free(hdc_qtable);
    if (knob::le_enable_trace && trace) {
        fclose(trace);
    }
}

uint32_t LearningEngineHDCBasic::chooseAction(EncodedState* encodedState) {
    stats.action.called++;

    uint32_t action = 0;
    if (m_action_counter < m_early_exploration_window) {
        action = (*actiongen)(generator);  // take random action
        stats.action.explore++;
        stats.action.dist[action][0]++;
        MYLOG("action taken %u explore,  scores %s", action, getStringQ(*encodedState).c_str());
        return action;  // fallback index
    }

    int best_idx = 0;  // Default fallback (offset 0)
    float best_sim = -2.0f;

    // 遍历所有动作
    for (size_t a = 0; a < m_a_.size(); ++a) {
        // 1. 临时二值化权重 M_a (Sign)
        // M_a[i] >= 0 -> Bit 0 (+1)
        // M_a[i] < 0  -> Bit 1 (-1)
        // 注意：这是为了和 HyperStream 的 HamminDistance 兼容
        BinaryHV m_bin;
        // 手动二值化 (这部分性能敏感，可以用 AVX2 优化，但先写标量版)
        for (size_t i = 0; i < HDC_DIM; ++i) {
            bool bit = (m_a_[a][i] < 0);  // 负数对应 bit 1
            m_bin.SetBit(i, bit);
        }

        // 2. 计算相似度 (Normalized Hamming 模拟 Cosine)
        // 你的 Python 是分别计算 sim_pc 和 sim_seq 取最大值
        NormalizedHammingArgs<HDC_DIM> args_pc{&m_bin, &encodedState->h_pc};
        float sim_pc = NormalizedHammingSimilarity(args_pc);

        NormalizedHammingArgs<HDC_DIM> args_seq{&m_bin, &encodedState->h_seq};
        float sim_seq = NormalizedHammingSimilarity(args_seq);

        float max_sim = std::max(sim_pc, sim_seq);
        hdc_qtable[a] = max_sim;

        if (max_sim > best_sim) {
            best_sim = max_sim;
            best_idx = a;
        }
    }

    // 3. 阈值判断 (THRESHOLD_BY_D logic)
    // D=2048 -> threshold ~ 0.088
    float threshold = knob::hdcpf_threshold;
    // cout << "best_sim:" << best_sim << endl;
    // cout << "best_idx:" << best_idx << endl;
    // cout << "threshold:" << threshold << endl;
    if (best_sim > threshold) {
        MYLOG("action taken %u exploit,  scores %s", best_idx, getStringQ(*encodedState).c_str());
        stats.action.exploit++;
        stats.action.dist[best_idx][1]++;
        return best_idx;
    } else {
        // auto it = action_to_index_map.find(0);
        // if (it != action_to_index_map.end()) {
        //     action = it->second;
        //     stats.action.explore++;
        //     stats.action.dist[action][0]++;
        //     MYLOG("action taken %u explore,  scores %s", action, getStringQ(*encodedState).c_str());
        // }

        action = (*actiongen)(generator);  // take random action
        stats.action.explore++;
        stats.action.dist[action][0]++;
        MYLOG("action taken %u explore,  scores %s", action, getStringQ(*encodedState).c_str());

        return action;  // fallback index
    }
}

// === 训练逻辑 (移植自 Python) ===
void LearningEngineHDCBasic::updateModel(EncodedState* old_state, uint32_t action, int32_t reward) {
    stats.learn.called++;
    stats.learn.dist[action][0]++;
    if (m_type == LearningType::SARSA && m_policy == Policy::EGreedy) {
        auto apply_update = [&](int action_idx, const BinaryHV& hv, int sign) {
            // sign = +1 (Add), sign = -1 (Sub)
            for (size_t i = 0; i < HDC_DIM; ++i) {
                // Binary 0 -> Value +1
                // Binary 1 -> Value -1
                int val = hv.GetBit(i) ? -1 : 1;

                int update = val * sign;
                int new_w = m_a_[action_idx][i] + update;

                // Clamp
                if (new_w > HDC_MAX_WEIGHT) new_w = HDC_MAX_WEIGHT;
                if (new_w < HDC_MIN_WEIGHT) new_w = HDC_MIN_WEIGHT;

                m_a_[action_idx][i] = (int8_t)new_w;
            }
        };
        // BinaryHV m_bin;
        // for (size_t i = 0; i < HDC_DIM; ++i) {
        //     bool bit = (m_a_[action][i] < 0);  // 负数对应 bit 1
        //     m_bin.SetBit(i, bit);
        // }
        // NormalizedHammingArgs<HDC_DIM> args_pc{&m_bin, &old_state->h_pc};
        // float sim_pc = NormalizedHammingSimilarity(args_pc);
        // NormalizedHammingArgs<HDC_DIM> args_seq{&m_bin, &old_state->h_seq};
        // float sim_seq = NormalizedHammingSimilarity(args_seq);

        // float threshold = knob::hdcpf_update_threshold;
        // if (sim_pc <= threshold) apply_update(action, old_state->h_pc, reward);
        // if (sim_seq <= threshold) apply_update(action, old_state->h_seq, reward);

        apply_update(action, old_state->h_pc, reward);
        apply_update(action, old_state->h_seq, reward);

    } else {
        printf("learning_type %s policy %s not supported!\n", MapLearningTypeString(m_type), MapPolicyString(m_policy));
        assert(false);
    }
}

void LearningEngineHDCBasic::updateModel_true(EncodedState* old_state, uint32_t action, int32_t reward) {
    stats.learnTrue.called++;
    stats.learnTrue.dist[action][0]++;
    MYLOG("learnTrue taken %u ", action);
    if (m_type == LearningType::SARSA && m_policy == Policy::EGreedy) {
        auto apply_update = [&](int action_idx, const BinaryHV& hv, int sign) {
            // sign = +1 (Add), sign = -1 (Sub)
            for (size_t i = 0; i < HDC_DIM; ++i) {
                // Binary 0 -> Value +1
                // Binary 1 -> Value -1
                int val = hv.GetBit(i) ? -1 : 1;

                int update = val * sign;
                int new_w = m_a_[action_idx][i] + update;

                // Clamp
                if (new_w > HDC_MAX_WEIGHT) new_w = HDC_MAX_WEIGHT;
                if (new_w < HDC_MIN_WEIGHT) new_w = HDC_MIN_WEIGHT;

                m_a_[action_idx][i] = (int8_t)new_w;
            }
        };
        // BinaryHV m_bin;
        // for (size_t i = 0; i < HDC_DIM; ++i) {
        //     bool bit = (m_a_[action][i] < 0);  // 负数对应 bit 1
        //     m_bin.SetBit(i, bit);
        // }
        // NormalizedHammingArgs<HDC_DIM> args_pc{&m_bin, &old_state->h_pc};
        // float sim_pc = NormalizedHammingSimilarity(args_pc);
        // NormalizedHammingArgs<HDC_DIM> args_seq{&m_bin, &old_state->h_seq};
        // float sim_seq = NormalizedHammingSimilarity(args_seq);

        // float threshold = knob::hdcpf_update_threshold;
        // if (sim_pc <= threshold) apply_update(action, old_state->h_pc, reward);
        // if (sim_seq <= threshold) apply_update(action, old_state->h_seq, reward);

        apply_update(action, old_state->h_pc, reward);
        apply_update(action, old_state->h_seq, reward);

    } else {
        printf("learning_type %s policy %s not supported!\n", MapLearningTypeString(m_type), MapPolicyString(m_policy));
        assert(false);
    }
}

uint32_t LearningEngineHDCBasic::getMaxAction(EncodedState* encodedState) {
    int best_idx = 0;  // Default fallback (offset 0)
    float best_sim = -1.0f;

    // 遍历所有动作
    for (size_t a = 0; a < m_a_.size(); ++a) {
        // 1. 临时二值化权重 M_a (Sign)
        // M_a[i] >= 0 -> Bit 0 (+1)
        // M_a[i] < 0  -> Bit 1 (-1)
        // 注意：这是为了和 HyperStream 的 HamminDistance 兼容
        BinaryHV m_bin;
        // 手动二值化 (这部分性能敏感，可以用 AVX2 优化，但先写标量版)
        for (size_t i = 0; i < HDC_DIM; ++i) {
            bool bit = (m_a_[a][i] < 0);  // 负数对应 bit 1
            m_bin.SetBit(i, bit);
        }

        // 2. 计算相似度 (Normalized Hamming 模拟 Cosine)
        // 你的 Python 是分别计算 sim_pc 和 sim_seq 取最大值
        NormalizedHammingArgs<HDC_DIM> args_pc{&m_bin, &encodedState->h_pc};
        float sim_pc = NormalizedHammingSimilarity(args_pc);

        NormalizedHammingArgs<HDC_DIM> args_seq{&m_bin, &encodedState->h_seq};
        float sim_seq = NormalizedHammingSimilarity(args_seq);

        float max_sim = std::max(sim_pc, sim_seq);
        hdc_qtable[a] = max_sim;

        if (max_sim > best_sim) {
            best_sim = max_sim;
            best_idx = a;
        }
    }

    // 3. 阈值判断 (THRESHOLD_BY_D logic)
    // D=2048 -> threshold ~ 0.088
    float threshold = 0.088f;
    if (best_sim > threshold) {
        return best_idx;
    } else {
        return 8;  // fallback index
    }
}

std::string LearningEngineHDCBasic::getStringQ(EncodedState state) {
    // assert(state < m_states);
    std::stringstream ss;
    for (uint32_t index = 0; index < m_actions; ++index) {
        ss << hdc_qtable[index] << ",";
    }
    return ss.str();
}

void LearningEngineHDCBasic::print_aux_stats() {
    /* compute state-action table usage
     * how mane state entries are actually used?
     * heat-map dump, etc. etc. */
    uint32_t state_used = 0;
    for (uint32_t state = 0; state < m_states; ++state) {
        for (uint32_t action = 0; action < m_actions; ++action) {
            if (hdc_qtable[action] != init_value) {
                state_used++;
                break;
            }
        }
    }

    fprintf(stdout, "learning_engine.state_used %u\n", state_used);
    fprintf(stdout, "\n");
}

void LearningEngineHDCBasic::dump_stats() {
    HDCPF* hdcpf = (HDCPF*)m_parent;
    fprintf(stdout, "learning_engine.action.called %lu\n", stats.action.called);
    fprintf(stdout, "learning_engine.action.explore %lu\n", stats.action.explore);
    fprintf(stdout, "learning_engine.action.exploit %lu\n", stats.action.exploit);
    for (uint32_t action = 0; action < m_actions; ++action) {
        fprintf(stdout, "learning_engine.action.index_%d_explored %lu\n", hdcpf->getAction(action), stats.action.dist[action][0]);
        fprintf(stdout, "learning_engine.action.index_%d_exploited %lu\n", hdcpf->getAction(action), stats.action.dist[action][1]);
    }
    fprintf(stdout, "learning_engine.learn.called %lu\n", stats.learn.called);
    for (uint32_t action = 0; action < m_actions; ++action) {
        fprintf(stdout, "learning_engine.action.index_%d_learn %lu\n", hdcpf->getAction(action), stats.learn.dist[action][0]);
    }
    fprintf(stdout, "learning_engine.learnTrue.called %lu\n", stats.learnTrue.called);
    for (uint32_t action = 0; action < m_actions; ++action) {
        fprintf(stdout, "learning_engine.action.index_%d_learnTrue %lu\n", hdcpf->getAction(action), stats.learnTrue.dist[action][0]);
    }

    fprintf(stdout, "\n");

    print_aux_stats();

    if (knob::le_enable_trace && knob::le_enable_score_plot) {
        plot_scores();
    }
}

void LearningEngineHDCBasic::dump_state_trace(uint32_t state) {
    trace_timestamp++;
    fprintf(trace, "%lu,", trace_timestamp);
    for (uint32_t index = 0; index < m_actions; ++index) {
        fprintf(trace, "%.2f,", hdc_qtable[index]);
    }
    fprintf(trace, "\n");
    fflush(trace);
}

void LearningEngineHDCBasic::plot_scores() {
    char* script_file = (char*)malloc(16 * sizeof(char));
    assert(script_file);
    gen_random(script_file, 16);
    FILE* script = fopen(script_file, "w");
    assert(script);

    fprintf(script, "set term png size 960,720 font 'Helvetica,12'\n");
    fprintf(script, "set datafile sep ','\n");
    fprintf(script, "set output '%s'\n", knob::le_plot_file_name.c_str());
    fprintf(script, "set title \"Reward over time\"\n");
    fprintf(script, "set xlabel \"Time\"\n");
    fprintf(script, "set ylabel \"Score\"\n");
    fprintf(script, "set grid y\n");
    fprintf(script, "set key right bottom Left box 3\n");
    fprintf(script, "plot ");
    for (uint32_t index = 0; index < knob::le_plot_actions.size(); ++index) {
        if (index) fprintf(script, ", ");
        fprintf(script, "'%s' using 1:%u with lines title \"action_index[%u]\"", knob::le_trace_file_name.c_str(), (knob::le_plot_actions[index] + 2), knob::le_plot_actions[index]);
    }
    fprintf(script, "\n");
    fclose(script);

    std::string cmd = "gnuplot " + std::string(script_file);
    system(cmd.c_str());

    std::string cmd2 = "rm " + std::string(script_file);
    system(cmd2.c_str());
}

void LearningEngineHDCBasic::dump_action_trace(uint32_t action) {
    action_trace_timestamp++;
    fprintf(action_trace, "%lu, %u\n", action_trace_timestamp, action);
}

// === 核心编码逻辑 (移植自 Python) ===
void LearningEngineHDCBasic::encode_state(
    EncodedState* encodedState,
    uint64_t pc, const std::vector<int32_t>& delta_pattern) {
    // --- 1. PC 部分 ---
    // Python: h_pc = bind(ROLE, bind(IM_PC[pc], IM_DELTA[last_delta]))
    int32_t last_delta = delta_pattern.empty() ? 0 : delta_pattern.back();

    BinaryHV pc_base = im_pc_[bucket_pc(pc)];
    BinaryHV delta_hv = im_delta_[bucket_delta(last_delta)];

    BinaryHV pc_delta_hv;
    Bind(pc_base, delta_hv, &pc_delta_hv);  // XOR Bind

    Bind(role_pc_lastdelta_, pc_delta_hv, &encodedState->h_pc);  // XOR Bind Role

    // --- 2. 序列部分 ---
    // Python: seq = bundle(permute(d_hv, i)...)
    // C++: 使用 BinaryBundler 来累加并做 Majority Vote

    BinaryBundler<HDC_DIM> bundler;  // 自动处理累加

    // reversed_deltas = delta_pattern[::-1] -> [new, ..., old]
    // 你的 Python 代码是 i, d in enumerate(reversed)，即 new 是 shift=0
    for (size_t i = 0; i < delta_pattern.size(); ++i) {
        // 取倒序元素
        int32_t d = delta_pattern[delta_pattern.size() - 1 - i];

        BinaryHV d_raw = im_delta_[bucket_delta(d)];
        BinaryHV d_permuted;

        // Permute (Rotate)
        PermuteRotate(d_raw, i, &d_permuted);

        // Accumulate
        bundler.Accumulate(d_permuted);
    }

    BinaryHV seq;
    bundler.Finalize(&seq);  // Majority Vote -> 变回 Binary

    Bind(role_delta_seq_, seq, &encodedState->h_seq);  // Bind Role

    // --- 3. 组合 (可选) ---
    // 你的 Python 代码里 bundle(h_pc, h_seq)
    // 这里我们简单用 Majority Vote (OR logic for binary approx? No, standard is majority)
    // 为了简单，我们也可以用 BundlePairMajority (类似 OR) 或者重新用 Bundler
    // 这里演示 Bundler:
    BinaryBundler<HDC_DIM> final_bundler;
    final_bundler.Accumulate(encodedState->h_pc);
    final_bundler.Accumulate(encodedState->h_seq);
    final_bundler.Finalize(&encodedState->combined);
}

// // === 训练逻辑 (移植自 Python) ===
// void LearningEngineHDC::train(const EncodedState& old_state, int true_action_idx, int old_action_idx) {
//     if (true_action_idx < 0 || true_action_idx >= (int)m_a_.size()) return;

//     // 1. 正样本加强 (Reinforce True Action)
//     // Python: M_a[true] += oldh_pc + oldh_seq
//     // 我们需要把 Binary 的 old_state 转换回 +1/-1 来加到 int8 权重上

//     auto apply_update = [&](int action_idx, const BinaryHV& hv, int sign) {
//         // sign = +1 (Add), sign = -1 (Sub)
//         for (size_t i = 0; i < HDC_DIM; ++i) {
//             // Binary 0 -> Value +1
//             // Binary 1 -> Value -1
//             int val = hv.GetBit(i) ? -1 : 1;

//             int update = val * sign;
//             int new_w = m_a_[action_idx][i] + update;

//             // Clamp
//             if (new_w > HDC_MAX_WEIGHT) new_w = HDC_MAX_WEIGHT;
//             if (new_w < HDC_MIN_WEIGHT) new_w = HDC_MIN_WEIGHT;

//             m_a_[action_idx][i] = (int8_t)new_w;
//         }
//     };

//     // 加强正确动作
//     // 注意：Python 代码里是 += oldh_pc 和 += oldh_seq
//     // 这里我们分开加，或者加 combined，取决于你的策略。
//     // 根据 Python: self.M_a[act] += oldh_pc; self.M_a[act] += oldh_seq;
//     apply_update(true_action_idx, old_state.h_pc, +1);
//     apply_update(true_action_idx, old_state.h_seq, +1);

//     // 2. 负样本抑制 (Punish Wrong Action)
//     if (old_action_idx != -1 && old_action_idx != true_action_idx) {
//         apply_update(old_action_idx, old_state.h_pc, -1);
//         apply_update(old_action_idx, old_state.h_seq, -1);
//     }
// }

// int LearningEngineHDC::get_action_index(int delta) {
//     auto it = delta_to_idx_map_.find(delta);
//     if (it != delta_to_idx_map_.end()) {
//         return it->second;
//     }
//     return -1;  // Not found
// }

// int LearningEngineHDC::get_action_delta(int index) {
//     if (index >= 0 && index < (int)ACTIONS_LIST.size()) {
//         return ACTIONS_LIST[index];
//     }
//     return 0;
// }