#include "hdc_pf.h"

#include <assert.h>

#include <algorithm>
#include <iomanip>
#include <iostream>

#include "cache.h"
#include "champsim.h"
#include "hdc_util.h"
#include "learning_engine_hdc_basic.h"
#include "memory_class.h"

// edit here
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

/* Action array
 * Basically a set of deltas to evaluate
 * Similar to the concept of BOP */
std::vector<int32_t> Actions;
std::unordered_map<int32_t, int32_t> action_to_index_map;

namespace knob {
extern std::size_t hdcpf_hdc_dim;
extern std::size_t hdcpf_hdc_n_pc;
extern std::size_t hdcpf_hdc_n_delta;
extern int hdcpf_hdc_max_weight;
extern int hdcpf_hdc_min_weight;
extern float hdcpf_threshold;
extern float hdcpf_update_threshold;
extern float hdcpf_alpha;
extern float hdcpf_gamma;
extern float hdcpf_epsilon;
extern uint32_t hdcpf_state_num_bits;
extern uint32_t hdcpf_max_states;
extern uint32_t hdcpf_seed;
extern string hdcpf_policy;
extern string hdcpf_learning_type;
extern vector<int32_t> hdcpf_actions;
extern uint32_t hdcpf_max_actions;
extern uint32_t hdcpf_pt_size;
extern uint32_t hdcpf_st_size;
extern uint32_t hdcpf_max_pcs;
extern uint32_t hdcpf_max_offsets;
extern uint32_t hdcpf_max_deltas;
extern int32_t hdcpf_reward_none;
extern int32_t hdcpf_reward_incorrect;
extern int32_t hdcpf_reward_correct_untimely;
extern int32_t hdcpf_reward_correct_timely;
extern bool hdcpf_brain_zero_init;
extern bool hdcpf_enable_reward_all;
extern bool hdcpf_enable_track_multiple;
extern bool hdcpf_enable_reward_out_of_bounds;
extern int32_t hdcpf_reward_out_of_bounds;
extern uint32_t hdcpf_state_type;
extern bool hdcpf_access_debug;
extern bool hdcpf_print_access_debug;
extern uint64_t hdcpf_print_access_debug_pc;
extern uint32_t hdcpf_print_access_debug_pc_count;
extern bool hdcpf_print_trace;
extern bool hdcpf_enable_state_action_stats;
extern bool hdcpf_enable_reward_tracker_hit;
extern int32_t hdcpf_reward_tracker_hit;
extern uint32_t hdcpf_state_hash_type;
extern bool hdcpf_enable_featurewise_engine;
extern uint32_t hdcpf_pref_degree;
extern bool hdcpf_enable_dyn_degree;
extern vector<float> hdcpf_max_to_avg_q_thresholds;
extern vector<int32_t> hdcpf_dyn_degrees;
extern uint64_t hdcpf_early_exploration_window;
extern uint32_t hdcpf_pt_address_hash_type;
extern uint32_t hdcpf_pt_address_hashed_bits;
extern uint32_t hdcpf_bloom_filter_size;
extern uint32_t hdcpf_multi_deg_select_type;
extern vector<int32_t> hdcpf_last_pref_offset_conf_thresholds;
extern vector<int32_t> hdcpf_dyn_degrees_type2;
extern uint32_t hdcpf_action_tracker_size;
extern uint32_t hdcpf_high_bw_thresh;
extern bool hdcpf_enable_hbw_reward;
extern int32_t hdcpf_reward_hbw_correct_timely;
extern int32_t hdcpf_reward_hbw_correct_untimely;
extern int32_t hdcpf_reward_hbw_incorrect;
extern int32_t hdcpf_reward_hbw_none;
extern int32_t hdcpf_reward_hbw_out_of_bounds;
extern int32_t hdcpf_reward_hbw_tracker_hit;

extern int32_t hdcpf_reward_hbw_notsure;
extern int32_t hdcpf_reward_notsure;

extern vector<int32_t> hdcpf_last_pref_offset_conf_thresholds_hbw;
extern vector<int32_t> hdcpf_dyn_degrees_type2_hbw;

/* Learning Engine knobs */
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

/* Featurewise Engine knobs */
extern vector<int32_t> hdcpf_le_featurewise_active_features;
extern vector<int32_t> hdcpf_le_featurewise_num_tilings;
extern vector<int32_t> hdcpf_le_featurewise_num_tiles;
extern vector<int32_t> hdcpf_le_featurewise_hash_types;
extern vector<int32_t> hdcpf_le_featurewise_enable_tiling_offset;
extern float hdcpf_le_featurewise_max_q_thresh;
extern bool hdcpf_le_featurewise_enable_action_fallback;
extern vector<float> hdcpf_le_featurewise_feature_weights;
extern bool hdcpf_le_featurewise_enable_dynamic_weight;
extern float hdcpf_le_featurewise_weight_gradient;
extern bool hdcpf_le_featurewise_disable_adjust_weight_all_features_align;
extern bool hdcpf_le_featurewise_selective_update;
extern uint32_t hdcpf_le_featurewise_pooling_type;
extern bool hdcpf_le_featurewise_enable_dyn_action_fallback;
extern uint32_t hdcpf_le_featurewise_bw_acc_check_level;
extern uint32_t hdcpf_le_featurewise_acc_thresh;
extern bool hdcpf_le_featurewise_enable_trace;
extern uint32_t hdcpf_le_featurewise_trace_feature_type;
extern string hdcpf_le_featurewise_trace_feature;
extern uint32_t hdcpf_le_featurewise_trace_interval;
extern uint32_t hdcpf_le_featurewise_trace_record_count;
extern std::string hdcpf_le_featurewise_trace_file_name;
extern bool hdcpf_le_featurewise_enable_score_plot;
extern vector<int32_t> hdcpf_le_featurewise_plot_actions;
extern std::string hdcpf_le_featurewise_plot_file_name;
extern bool hdcpf_le_featurewise_remove_plot_script;
}  // namespace knob

void HDCPF::init_knobs() {
    Actions.resize(knob::hdcpf_max_actions, 0);
    std::copy(knob::hdcpf_actions.begin(), knob::hdcpf_actions.end(), Actions.begin());
    assert(Actions.size() == knob::hdcpf_max_actions);
    assert(Actions.size() <= MAX_ACTIONS);
    // --- 新增代码 Start ---
    // 构建反向查找表：Key = Delta值, Value = Actions中的下标
    action_to_index_map.clear();
    for (size_t i = 0; i < Actions.size(); ++i) {
        action_to_index_map[Actions[i]] = (int32_t)i;
    }
    // --- 新增代码 End ---

    if (knob::hdcpf_access_debug) {
        cout << "***WARNING*** setting knob::hdcpf_max_pcs, knob::hdcpf_max_offsets, and knob::hdcpf_max_deltas to large value as knob::hdcpf_access_debug is true" << endl;
        knob::hdcpf_max_pcs = 1024;
        knob::hdcpf_max_offsets = 1024;
        knob::hdcpf_max_deltas = 1024;
    }
    assert(knob::hdcpf_pref_degree >= 1 && (knob::hdcpf_pref_degree == 1 || !knob::hdcpf_enable_dyn_degree));
    assert(knob::hdcpf_max_to_avg_q_thresholds.size() == knob::hdcpf_dyn_degrees.size() - 1);
    assert(knob::hdcpf_last_pref_offset_conf_thresholds.size() == knob::hdcpf_dyn_degrees_type2.size() - 1);
}

void HDCPF::init_stats() {
    bzero(&stats, sizeof(stats));
    stats.predict.action_dist.resize(knob::hdcpf_max_actions, 0);
    stats.predict.issue_dist.resize(knob::hdcpf_max_actions, 0);
    stats.predict.pred_hit.resize(knob::hdcpf_max_actions, 0);
    stats.predict.out_of_bounds_dist.resize(knob::hdcpf_max_actions, 0);
    state_action_dist.clear();
}

HDCPF::HDCPF(string type) : Prefetcher(type) {
    init_knobs();
    init_stats();

    recorder = new HDCPFRecorder();

    last_evicted_tracker = NULL;

    /* init learning engine */

    brain_hdc_basic = NULL;

    brain_hdc_basic = new LearningEngineHDCBasic(this,
                                                 knob::hdcpf_alpha, knob::hdcpf_gamma, knob::hdcpf_epsilon,
                                                 knob::hdcpf_max_actions,
                                                 knob::hdcpf_max_states,
                                                 knob::hdcpf_seed,
                                                 knob::hdcpf_policy,
                                                 knob::hdcpf_learning_type,
                                                 knob::hdcpf_brain_zero_init,
                                                 knob::hdcpf_early_exploration_window);

    bw_level = 0;
    core_ipc = 0;
}

HDCPF::~HDCPF() {
    // if (brain_featurewise) delete brain_featurewise;
    // if (brain) delete brain;
    if (brain_hdc_basic) delete brain_hdc_basic;
}

void HDCPF::print_config() {
    cout << "hdcpf_alpha " << knob::hdcpf_alpha << endl
         << "hdcpf_gamma " << knob::hdcpf_gamma << endl
         << "hdcpf_epsilon " << knob::hdcpf_epsilon << endl
         << "hdcpf_state_num_bits " << knob::hdcpf_state_num_bits << endl
         << "hdcpf_max_states " << knob::hdcpf_max_states << endl
         << "hdcpf_seed " << knob::hdcpf_seed << endl
         << "hdcpf_policy " << knob::hdcpf_policy << endl
         << "hdcpf_learning_type " << knob::hdcpf_learning_type << endl
         << "hdcpf_actions " << array_to_string(Actions) << endl
         << "hdcpf_max_actions " << knob::hdcpf_max_actions << endl
         << "hdcpf_pt_size " << knob::hdcpf_pt_size << endl
         << "hdcpf_st_size " << knob::hdcpf_st_size << endl
         << "hdcpf_max_pcs " << knob::hdcpf_max_pcs << endl
         << "hdcpf_max_offsets " << knob::hdcpf_max_offsets << endl
         << "hdcpf_max_deltas " << knob::hdcpf_max_deltas << endl
         << "hdcpf_reward_none " << knob::hdcpf_reward_none << endl
         << "hdcpf_reward_incorrect " << knob::hdcpf_reward_incorrect << endl
         << "hdcpf_reward_correct_untimely " << knob::hdcpf_reward_correct_untimely << endl
         << "hdcpf_reward_correct_timely " << knob::hdcpf_reward_correct_timely << endl
         << "hdcpf_brain_zero_init " << knob::hdcpf_brain_zero_init << endl
         << "hdcpf_enable_reward_all " << knob::hdcpf_enable_reward_all << endl
         << "hdcpf_enable_track_multiple " << knob::hdcpf_enable_track_multiple << endl
         << "hdcpf_enable_reward_out_of_bounds " << knob::hdcpf_enable_reward_out_of_bounds << endl
         << "hdcpf_reward_out_of_bounds " << knob::hdcpf_reward_out_of_bounds << endl
         << "hdcpf_state_type " << knob::hdcpf_state_type << endl
         << "hdcpf_state_hash_type " << knob::hdcpf_state_hash_type << endl
         << "hdcpf_access_debug " << knob::hdcpf_access_debug << endl
         << "hdcpf_print_access_debug " << knob::hdcpf_print_access_debug << endl
         << "hdcpf_print_access_debug_pc " << hex << knob::hdcpf_print_access_debug_pc << dec << endl
         << "hdcpf_print_access_debug_pc_count " << knob::hdcpf_print_access_debug_pc_count << endl
         << "hdcpf_print_trace " << knob::hdcpf_print_trace << endl
         << "hdcpf_enable_state_action_stats " << knob::hdcpf_enable_state_action_stats << endl
         << "hdcpf_enable_reward_tracker_hit " << knob::hdcpf_enable_reward_tracker_hit << endl
         << "hdcpf_reward_tracker_hit " << knob::hdcpf_reward_tracker_hit << endl
         << "hdcpf_enable_featurewise_engine " << knob::hdcpf_enable_featurewise_engine << endl
         << "hdcpf_pref_degree " << knob::hdcpf_pref_degree << endl
         << "hdcpf_enable_dyn_degree " << knob::hdcpf_enable_dyn_degree << endl
         << "hdcpf_max_to_avg_q_thresholds " << array_to_string(knob::hdcpf_max_to_avg_q_thresholds) << endl
         << "hdcpf_dyn_degrees " << array_to_string(knob::hdcpf_dyn_degrees) << endl
         << "hdcpf_multi_deg_select_type " << knob::hdcpf_multi_deg_select_type << endl
         << "hdcpf_last_pref_offset_conf_thresholds " << array_to_string(knob::hdcpf_last_pref_offset_conf_thresholds) << endl
         << "hdcpf_dyn_degrees_type2 " << array_to_string(knob::hdcpf_dyn_degrees_type2) << endl
         << "hdcpf_action_tracker_size " << knob::hdcpf_action_tracker_size << endl
         << "hdcpf_high_bw_thresh " << knob::hdcpf_high_bw_thresh << endl
         << "hdcpf_enable_hbw_reward " << knob::hdcpf_enable_hbw_reward << endl
         << "hdcpf_reward_hbw_correct_timely " << knob::hdcpf_reward_hbw_correct_timely << endl
         << "hdcpf_reward_hbw_correct_untimely " << knob::hdcpf_reward_hbw_correct_untimely << endl
         << "hdcpf_reward_hbw_incorrect " << knob::hdcpf_reward_hbw_incorrect << endl
         << "hdcpf_reward_hbw_none " << knob::hdcpf_reward_hbw_none << endl
         << "hdcpf_reward_hbw_out_of_bounds " << knob::hdcpf_reward_hbw_out_of_bounds << endl
         << "hdcpf_reward_hbw_tracker_hit " << knob::hdcpf_reward_hbw_tracker_hit << endl
         << "hdcpf_last_pref_offset_conf_thresholds_hbw " << array_to_string(knob::hdcpf_last_pref_offset_conf_thresholds_hbw) << endl
         << "hdcpf_dyn_degrees_type2_hbw " << array_to_string(knob::hdcpf_dyn_degrees_type2_hbw) << endl
         << endl
         << "le_enable_trace " << knob::le_enable_trace << endl
         << "le_trace_interval " << knob::le_trace_interval << endl
         << "le_trace_file_name " << knob::le_trace_file_name << endl
         << "le_trace_state " << hex << knob::le_trace_state << dec << endl
         << "le_enable_score_plot " << knob::le_enable_score_plot << endl
         << "le_plot_file_name " << knob::le_plot_file_name << endl
         << "le_plot_actions " << array_to_string(knob::le_plot_actions) << endl
         << "le_enable_action_trace " << knob::le_enable_action_trace << endl
         << "le_action_trace_interval " << knob::le_action_trace_interval << endl
         << "le_action_trace_name " << knob::le_action_trace_name << endl
         << "le_enable_action_plot " << knob::le_enable_action_plot << endl
         << endl
         << "hdcpf_le_featurewise_active_features " << print_active_features2(knob::hdcpf_le_featurewise_active_features) << endl
         << "hdcpf_le_featurewise_num_tilings " << array_to_string(knob::hdcpf_le_featurewise_num_tilings) << endl
         << "hdcpf_le_featurewise_num_tiles " << array_to_string(knob::hdcpf_le_featurewise_num_tiles) << endl
         << "hdcpf_le_featurewise_hash_types " << array_to_string(knob::hdcpf_le_featurewise_hash_types) << endl
         << "hdcpf_le_featurewise_enable_tiling_offset " << array_to_string(knob::hdcpf_le_featurewise_enable_tiling_offset) << endl
         << "hdcpf_le_featurewise_max_q_thresh " << knob::hdcpf_le_featurewise_max_q_thresh << endl
         << "hdcpf_le_featurewise_enable_action_fallback " << knob::hdcpf_le_featurewise_enable_action_fallback << endl
         << "hdcpf_le_featurewise_feature_weights " << array_to_string(knob::hdcpf_le_featurewise_feature_weights) << endl
         << "hdcpf_le_featurewise_enable_dynamic_weight " << knob::hdcpf_le_featurewise_enable_dynamic_weight << endl
         << "hdcpf_le_featurewise_weight_gradient " << knob::hdcpf_le_featurewise_weight_gradient << endl
         << "hdcpf_le_featurewise_disable_adjust_weight_all_features_align " << knob::hdcpf_le_featurewise_disable_adjust_weight_all_features_align << endl
         << "hdcpf_le_featurewise_selective_update " << knob::hdcpf_le_featurewise_selective_update << endl
         << "hdcpf_le_featurewise_pooling_type " << knob::hdcpf_le_featurewise_pooling_type << endl
         << "hdcpf_le_featurewise_enable_dyn_action_fallback " << knob::hdcpf_le_featurewise_enable_dyn_action_fallback << endl
         << "hdcpf_le_featurewise_bw_acc_check_level " << knob::hdcpf_le_featurewise_bw_acc_check_level << endl
         << "hdcpf_le_featurewise_acc_thresh " << knob::hdcpf_le_featurewise_acc_thresh << endl
         << "hdcpf_le_featurewise_enable_trace " << knob::hdcpf_le_featurewise_enable_trace << endl
         << "hdcpf_le_featurewise_trace_feature_type " << knob::hdcpf_le_featurewise_trace_feature_type << endl
         << "hdcpf_le_featurewise_trace_feature " << knob::hdcpf_le_featurewise_trace_feature << endl
         << "hdcpf_le_featurewise_trace_interval " << knob::hdcpf_le_featurewise_trace_interval << endl
         << "hdcpf_le_featurewise_trace_record_count " << knob::hdcpf_le_featurewise_trace_record_count << endl
         << "hdcpf_le_featurewise_trace_file_name " << knob::hdcpf_le_featurewise_trace_file_name << endl
         << "hdcpf_le_featurewise_enable_score_plot " << knob::hdcpf_le_featurewise_enable_score_plot << endl
         << "hdcpf_le_featurewise_plot_actions " << array_to_string(knob::hdcpf_le_featurewise_plot_actions) << endl
         << "hdcpf_le_featurewise_plot_file_name " << knob::hdcpf_le_featurewise_plot_file_name << endl
         << endl;
}

void HDCPF::invoke_prefetcher(uint64_t pc, uint64_t address, uint8_t cache_hit, uint8_t type, vector<uint64_t>& pref_addr) {
    uint64_t page = address >> LOG2_PAGE_SIZE;
    uint32_t offset = (address >> LOG2_BLOCK_SIZE) & ((1ull << (LOG2_PAGE_SIZE - LOG2_BLOCK_SIZE)) - 1);

    MYLOG("---------------------------------------------------------------------");
    MYLOG("%s %lx pc %lx page %lx off %u", GetAccessType(type), address, pc, page, offset);

    /* compute reward on demand */
    reward(address);

    /* record the access: just to gain some insights from the workload
     * defined in hdcpf_helper.h(cc) */
    recorder->record_access(pc, address, page, offset, bw_level);

    /* global state tracking */
    update_global_state(pc, page, offset, address);
    /* per page state tracking */
    HDCPF_STEntry* stentry = update_local_state(pc, page, offset, address);

    /* Measure state.
     * state can contain per page local information like delta signature, pc signature etc.
     * it can also contain global signatures like last three branch PCs etc.
     */
    State* state = new State();
    state->pc = pc;
    state->address = address;
    state->page = page;
    state->offset = offset;
    state->delta = !stentry->deltas.empty() ? stentry->deltas.back() : 0;
    state->local_delta_sig = stentry->get_delta_sig();
    state->local_delta_sig2 = stentry->get_delta_sig2();
    state->local_pc_sig = stentry->get_pc_sig();
    state->local_offset_sig = stentry->get_offset_sig();
    state->bw_level = bw_level;
    state->is_high_bw = is_high_bw();
    state->acc_level = acc_level;

    // 检查一下是否需要学习
    if (stentry->learnByNextDelta) {
        if (state->delta != 0) {
            auto it = action_to_index_map.find(state->delta);
            if (it != action_to_index_map.end()) {
                brain_hdc_basic->updateModel_true(&(stentry->eState), it->second, 1);
            }
        }
        stentry->learnByNextDelta = false;
    }

    uint32_t count = pref_addr.size();
    // 编码器
    encodedState = new EncodedState();

    // delta序列
    std::vector<int32_t> delta_pattern;
    delta_pattern.reserve(4);
    if (stentry->deltas.size() >= 4) {
        for (size_t i = stentry->deltas.size() - 4; i < stentry->deltas.size(); ++i) {
            delta_pattern.push_back(stentry->deltas[i]);
        }

    } else {
        for (size_t i = 0; i < stentry->deltas.size(); ++i) {
            delta_pattern.push_back(stentry->deltas[i]);
        }
        for (size_t i = stentry->deltas.size() - 4; i < 4; ++i) {
            delta_pattern.push_back(0);
        }
    }
    brain_hdc_basic->encode_state(encodedState, pc, delta_pattern);
    // MYLOG("addr@%lx page %lx off %u state %x", encodedState->h_pc,encodedState->h_se,encodedState->combined);
    // // 假设 encodedState 是指针
    // const auto& words = encodedState->h_pc.Words();  // 获取底层 uint64_t 数组
    // std::cout << "h_pc (Hex): ";
    // for (const auto& w : words) {
    //     // 打印为 16 位宽度的 16 进制，前面补 0
    //     std::cout << std::hex << std::setw(16) << std::setfill('0') << w << " ";
    // }
    // const auto& words1 = encodedState->h_seq.Words();  // 获取底层 uint64_t 数组
    // std::cout << "h_seq (Hex): ";
    // for (const auto& w : words1) {
    //     // 打印为 16 位宽度的 16 进制，前面补 0
    //     std::cout << std::hex << std::setw(16) << std::setfill('0') << w << " ";
    // }
    // std::cout << std::dec << std::endl;  // 恢复十进制

    predict(address, page, offset, state, pref_addr, encodedState);
    stats.pref_issue.hdcpf += (pref_addr.size() - count);
}

void HDCPF::update_global_state(uint64_t pc, uint64_t page, uint32_t offset, uint64_t address) {
    /* @rbera TODO: implement */
}

HDCPF_STEntry* HDCPF::update_local_state(uint64_t pc, uint64_t page, uint32_t offset, uint64_t address) {
    stats.st.lookup++;
    HDCPF_STEntry* stentry = NULL;
    auto st_index = find_if(signature_table.begin(), signature_table.end(), [page](HDCPF_STEntry* stentry) { return stentry->page == page; });
    if (st_index != signature_table.end()) {
        stats.st.hit++;
        stentry = (*st_index);
        stentry->update(page, pc, offset, address);
        signature_table.erase(st_index);
        signature_table.push_back(stentry);
        return stentry;
    } else {
        if (signature_table.size() >= knob::hdcpf_st_size) {
            stats.st.evict++;
            stentry = signature_table.front();
            signature_table.pop_front();
            if (knob::hdcpf_access_debug) {
                recorder->record_access_knowledge(stentry);
                if (knob::hdcpf_print_access_debug) {
                    print_access_debug(stentry);
                }
            }
            delete stentry;
        }

        stats.st.insert++;
        stentry = new HDCPF_STEntry(page, pc, offset);
        recorder->record_trigger_access(page, pc, offset);
        signature_table.push_back(stentry);
        return stentry;
    }
}

uint32_t HDCPF::predict(uint64_t base_address, uint64_t page, uint32_t offset, State* state, vector<uint64_t>& pref_addr, EncodedState* encodedState) {
    MYLOG("addr@%lx page %lx off %u state %x", base_address, page, offset, state->value());

    stats.predict.called++;

    /* query learning engine to get the next prediction */
    uint32_t action_index = 0;
    uint32_t pref_degree = knob::hdcpf_pref_degree;
    vector<bool> consensus_vec;  // only required for featurewise engine

    action_index = brain_hdc_basic->chooseAction(encodedState);

    if (knob::hdcpf_enable_state_action_stats) {
        update_stats(state, action_index, pref_degree);
    }

    assert(action_index < knob::hdcpf_max_actions);

    MYLOG("act_idx %u act %d", action_index, Actions[action_index]);

    uint64_t addr = 0xdeadbeef;
    HDCPF_PTEntry* ptentry = NULL;
    int32_t predicted_offset = 0;
    if (Actions[action_index] != 0) {
        predicted_offset = (int32_t)offset + Actions[action_index];
        if (predicted_offset >= 0 && predicted_offset < 64) /* falls within the page */
        {
            int64_t offset_in_bytes = (int64_t)predicted_offset << LOG2_BLOCK_SIZE;
            addr = (page << LOG2_PAGE_SIZE) + offset_in_bytes;

            // addr = (page << LOG2_PAGE_SIZE) + (predicted_offset << LOG2_BLOCK_SIZE);
            MYLOG("pred_off %d pred_addr %lx", predicted_offset, addr);
            /* track prefetch */
            bool new_addr = track(addr, base_address, state, action_index, &ptentry, encodedState);
            if (new_addr) {
                pref_addr.push_back(addr);
                track_in_st(page, predicted_offset, Actions[action_index]);
                stats.predict.issue_dist[action_index]++;
                if (pref_degree > 1) {
                    gen_multi_degree_pref(page, offset, Actions[action_index], pref_degree, pref_addr);
                }
                stats.predict.deg_histogram[pref_degree]++;
                ptentry->consensus_vec = consensus_vec;
            } else {
                MYLOG("pred_off %d tracker_hit", predicted_offset);
                stats.predict.pred_hit[action_index]++;
                if (knob::hdcpf_enable_reward_tracker_hit) {
                    addr = 0xdeadbeef;
                    track(addr, base_address, state, action_index, &ptentry, encodedState);
                    assert(ptentry);
                    assign_reward(ptentry, RewardType::tracker_hit);
                    ptentry->consensus_vec = consensus_vec;
                }
            }
            stats.predict.action_dist[action_index]++;
        } else {
            MYLOG("pred_off %d out_of_bounds", predicted_offset);
            stats.predict.out_of_bounds++;
            stats.predict.out_of_bounds_dist[action_index]++;
            if (knob::hdcpf_enable_reward_out_of_bounds) {
                addr = 0xdeadbeef;
                track(addr, base_address, state, action_index, &ptentry, encodedState);
                assert(ptentry);
                assign_reward(ptentry, RewardType::out_of_bounds);
                ptentry->consensus_vec = consensus_vec;
            }
        }
    } else {
        MYLOG("no prefecth");
        /* agent decided not to prefetch */
        addr = 0xdeadbeef;
        /* track no prefetch */
        track(addr, base_address, state, action_index, &ptentry, encodedState);
        stats.predict.action_dist[action_index]++;
        ptentry->consensus_vec = consensus_vec;
    }

    stats.predict.predicted += pref_addr.size();
    MYLOG("end@%lx", base_address);

    return pref_addr.size();
}

/* Returns true if the address is not already present in prefetch_tracker
 * false otherwise */
bool HDCPF::track(uint64_t address, uint64_t address_trigger, State* state, uint32_t action_index, HDCPF_PTEntry** tracker, EncodedState* encodedState) {
    MYLOG("addr@%lx state %x act_idx %u act %d", address, state->value(), action_index, Actions[action_index]);
    stats.track.called++;

    bool new_addr = true;
    vector<HDCPF_PTEntry*> ptentries = search_pt(address, false);
    if (ptentries.empty()) {
        new_addr = true;
    } else {
        new_addr = false;
    }

    if (!new_addr && address != 0xdeadbeef && !knob::hdcpf_enable_track_multiple) {
        stats.track.same_address++;
        tracker = NULL;
        return new_addr;
    }

    /* new prefetched address that hasn't been seen before */
    HDCPF_PTEntry* ptentry = NULL;

    if (prefetch_tracker.size() >= knob::hdcpf_pt_size) {
        stats.track.evict++;
        ptentry = prefetch_tracker.front();
        prefetch_tracker.pop_front();
        MYLOG("victim_state %x victim_act_idx %u victim_act %d", ptentry->state->value(), ptentry->action_index, Actions[ptentry->action_index]);
        if (last_evicted_tracker) {
            MYLOG("last_victim_state %x last_victim_act_idx %u last_victim_act %d", last_evicted_tracker->state->value(), last_evicted_tracker->action_index, Actions[last_evicted_tracker->action_index]);
            /* train the agent */
            train(ptentry, last_evicted_tracker);
            delete last_evicted_tracker->state;
            delete last_evicted_tracker->encodedState;
            delete last_evicted_tracker;
        }
        last_evicted_tracker = ptentry;
    }

    ptentry = new HDCPF_PTEntry(address, address_trigger, state, action_index, encodedState);
    prefetch_tracker.push_back(ptentry);
    assert(prefetch_tracker.size() <= knob::hdcpf_pt_size);

    (*tracker) = ptentry;
    MYLOG("end@%lx", address);

    return new_addr;
}

void HDCPF::gen_multi_degree_pref(uint64_t page, uint32_t offset, int32_t action, uint32_t pref_degree, vector<uint64_t>& pref_addr) {
    stats.predict.multi_deg_called++;
    uint64_t addr = 0xdeadbeef;
    int32_t predicted_offset = 0;
    if (action != 0) {
        for (uint32_t degree = 2; degree <= pref_degree; ++degree) {
            predicted_offset = (int32_t)offset + degree * action;
            if (predicted_offset >= 0 && predicted_offset < 64) {
                int64_t offset_in_bytes = (int64_t)predicted_offset << LOG2_BLOCK_SIZE;
                addr = (page << LOG2_PAGE_SIZE) + offset_in_bytes;
                // addr = (page << LOG2_PAGE_SIZE) + (predicted_offset << LOG2_BLOCK_SIZE);
                pref_addr.push_back(addr);
                MYLOG("degree %u pred_off %d pred_addr %lx", degree, predicted_offset, addr);
                stats.predict.multi_deg++;
                stats.predict.multi_deg_histogram[degree]++;
            }
        }
    }
}

uint32_t HDCPF::get_dyn_pref_degree(float max_to_avg_q_ratio, uint64_t page, int32_t action) {
    uint32_t counted = false;
    uint32_t degree = 1;

    if (knob::hdcpf_multi_deg_select_type == 2) {
        auto st_index = find_if(signature_table.begin(), signature_table.end(), [page](HDCPF_STEntry* stentry) { return stentry->page == page; });
        if (st_index != signature_table.end()) {
            int32_t conf = 0;
            bool found = (*st_index)->search_action_tracker(action, conf);
            vector<int32_t> conf_thresholds, deg_afterburning, deg_normal;

            conf_thresholds = is_high_bw() ? knob::hdcpf_last_pref_offset_conf_thresholds_hbw : knob::hdcpf_last_pref_offset_conf_thresholds;
            deg_normal = is_high_bw() ? knob::hdcpf_dyn_degrees_type2_hbw : knob::hdcpf_dyn_degrees_type2;

            if (found) {
                for (uint32_t index = 0; index < conf_thresholds.size(); ++index) {
                    /* hdcpf_last_pref_offset_conf_thresholds is a sorted list in ascending order of values */
                    if (conf <= conf_thresholds[index]) {
                        degree = deg_normal[index];
                        counted = true;
                        break;
                    }
                }
                if (!counted) {
                    degree = deg_normal.back();
                }
            } else {
                degree = 1;
            }
        }
    }
    return degree;
}

/* This reward fucntion is called after seeing a demand access to the address */
/* TODO: what if multiple prefetch request generated the same address?
 * Currently, it just rewards the oldest prefetch request to the address
 * Should we reward all? */
void HDCPF::reward(uint64_t address) {
    MYLOG("addr @ %lx", address);
    /*start标记真实标签*/
    uint64_t page = address >> LOG2_PAGE_SIZE;
    uint32_t offset = (address >> LOG2_BLOCK_SIZE) & ((1ull << (LOG2_PAGE_SIZE - LOG2_BLOCK_SIZE)) - 1);
    vector<HDCPF_PTEntry*> ptentries_insamepage = search_pt_by_page(address, knob::hdcpf_enable_reward_all);
    for (auto* entry : ptentries_insamepage) {
        // 计算真实的 delta
        // 假设 HDCPF_PTEntry 中有 offset_trigger 字段 (根据你的描述)
        // 如果只有 address_trigger，则需要先计算:
        // int32_t trigger_offset = (entry->address_trigger >> LOG2_BLOCK_SIZE) & ((1ull << (LOG2_PAGE_SIZE - LOG2_BLOCK_SIZE)) - 1);
        // 如果还没有fill,就更新,保留最靠近的
        if (!entry->true_action_valid) {
            int32_t true_delta = (int32_t)offset - (int32_t)entry->offset_trigger;
            // 在映射表中查找是否存在这个 delta
            if (true_delta != 0) {
                auto it = action_to_index_map.find(true_delta);
                if (it != action_to_index_map.end()) {
                    // 找到了，设置 index
                    entry->true_action = it->second;
                    entry->true_action_valid = true;
                    // 可选：打印日志调试
                    // MYLOG("True Label Found: trigger_off=%u, curr_off=%u, delta=%d, action_idx=%d",
                    //       entry->offset_trigger, current_offset, true_delta, it->second);
                }
            }
        }
    }
    /*end标记真实标签*/

    stats.reward.demand.called++;
    vector<HDCPF_PTEntry*> ptentries = search_pt(address, knob::hdcpf_enable_reward_all);

    if (ptentries.empty()) {
        MYLOG("PT miss");
        stats.reward.demand.pt_not_found++;
        return;
    } else {
        stats.reward.demand.pt_found++;
    }

    for (uint32_t index = 0; index < ptentries.size(); ++index) {
        // todo cyh here
        HDCPF_PTEntry* ptentry = ptentries[index];
        stats.reward.demand.pt_found_total++;

        MYLOG("PT hit. state %x act_idx %u act %d", ptentry->state->value(), ptentry->action_index, Actions[ptentry->action_index]);
        /* Do not compute reward if already has a reward.
         * This can happen when a prefetch access sees multiple demand reuse */
        if (ptentry->has_reward) {
            MYLOG("entry already has reward: %d", ptentry->reward);
            stats.reward.demand.has_reward++;
            return;
        }

        if (ptentry->is_filled) /* timely */
        {
            assign_reward(ptentry, RewardType::correct_timely);
            MYLOG("assigned reward correct_timely(%d)", ptentry->reward);
        } else {
            assign_reward(ptentry, RewardType::correct_untimely);
            MYLOG("assigned reward correct_untimely(%d)", ptentry->reward);
        }
        ptentry->has_reward = true;
    }
}

/* This reward function is called during eviction from prefetch_tracker */
void HDCPF::reward(HDCPF_PTEntry* ptentry) {
    MYLOG("reward PT evict %lx state %x act_idx %u act %d", ptentry->address, ptentry->state->value(), ptentry->action_index, Actions[ptentry->action_index]);

    stats.reward.train.called++;
    assert(!ptentry->has_reward);
    /* this is called during eviction from prefetch tracker
     * that means, this address doesn't see a demand reuse.
     * hence it either can be incorrect, or no prefetch */
    if (ptentry->address == 0xdeadbeef) /* no prefetch */
    {
        assign_reward(ptentry, RewardType::none);
        MYLOG("assigned reward no_pref(%d)", ptentry->reward);
    } else /* incorrect prefetch */
    {
        if (ptentry->true_action_valid) {
            assign_reward(ptentry, RewardType::incorrect);
            MYLOG("assigned reward incorrect(%d)", ptentry->reward);
        } else {
            assign_reward(ptentry, RewardType::notsure);
            MYLOG("assigned reward notsure(%d)", ptentry->reward);
            // 这里要更新一下stentry,让准备下一次访问到来的时候,学习一下;

            uint64_t page = ptentry->page_trigger;
            HDCPF_STEntry* stentry = NULL;
            auto st_index = find_if(signature_table.begin(), signature_table.end(), [page](HDCPF_STEntry* stentry) { return stentry->page == page; });
            if (st_index != signature_table.end()) {
                // (*st_index)->track_prefetch(pred_offset, pref_offset);
                (*st_index)->learnByNextDelta = true;
                (*st_index)->eState.h_pc = ptentry->encodedState->h_pc;
                (*st_index)->eState.h_seq = ptentry->encodedState->h_seq;
                (*st_index)->eState.combined = ptentry->encodedState->combined;
            }
        }
    }
    ptentry->has_reward = true;
}

void HDCPF::assign_reward(HDCPF_PTEntry* ptentry, RewardType type) {
    MYLOG("assign_reward PT evict %lx state %x act_idx %u act %d", ptentry->address, ptentry->state->value(), ptentry->action_index, Actions[ptentry->action_index]);
    assert(!ptentry->has_reward);

    /* compute the reward */
    int32_t reward = compute_reward(ptentry, type);

    /* assign */
    ptentry->reward = reward;
    ptentry->reward_type = type;
    ptentry->has_reward = true;

    /* maintain stats */
    stats.reward.assign_reward.called++;
    switch (type) {
        case RewardType::correct_timely:
            stats.reward.correct_timely++;
            break;
        case RewardType::correct_untimely:
            stats.reward.correct_untimely++;
            break;
        case RewardType::incorrect:
            stats.reward.incorrect++;
            break;
        case RewardType::none:
            stats.reward.no_pref++;
            break;
        case RewardType::out_of_bounds:
            stats.reward.out_of_bounds++;
            break;
        case RewardType::tracker_hit:
            stats.reward.tracker_hit++;
            break;
        case RewardType::notsure:
            stats.reward.notsure++;
            break;
        default:
            assert(false);
    }
    stats.reward.dist[ptentry->action_index][type]++;
}

int32_t HDCPF::compute_reward(HDCPF_PTEntry* ptentry, RewardType type) {
    bool high_bw = (knob::hdcpf_enable_hbw_reward && is_high_bw()) ? true : false;
    int32_t reward = 0;

    stats.reward.compute_reward.dist[type][high_bw]++;

    if (type == RewardType::correct_timely) {
        reward = high_bw ? knob::hdcpf_reward_hbw_correct_timely : knob::hdcpf_reward_correct_timely;
    } else if (type == RewardType::correct_untimely) {
        reward = high_bw ? knob::hdcpf_reward_hbw_correct_untimely : knob::hdcpf_reward_correct_untimely;
    } else if (type == RewardType::incorrect) {
        reward = high_bw ? knob::hdcpf_reward_hbw_incorrect : knob::hdcpf_reward_incorrect;
    } else if (type == RewardType::none) {
        reward = high_bw ? knob::hdcpf_reward_hbw_none : knob::hdcpf_reward_none;
    } else if (type == RewardType::out_of_bounds) {
        reward = high_bw ? knob::hdcpf_reward_hbw_out_of_bounds : knob::hdcpf_reward_out_of_bounds;
    } else if (type == RewardType::tracker_hit) {
        reward = high_bw ? knob::hdcpf_reward_hbw_tracker_hit : knob::hdcpf_reward_tracker_hit;
    } else if (type == RewardType::notsure) {
        reward = high_bw ? knob::hdcpf_reward_hbw_notsure : knob::hdcpf_reward_notsure;
    }

    else {
        cout << "Invalid reward type found " << type << endl;
        assert(false);
    }

    // if (type == RewardType::correct_timely) {
    //     reward = high_bw ? knob::hdcpf_reward_hbw_correct_timely : knob::hdcpf_reward_correct_timely;
    // } else if (type == RewardType::correct_untimely) {
    //     reward = high_bw ? knob::hdcpf_reward_hbw_correct_untimely : knob::hdcpf_reward_correct_untimely;
    // } else if (type == RewardType::incorrect) {
    //     reward = high_bw ? knob::hdcpf_reward_hbw_incorrect : knob::hdcpf_reward_incorrect;
    // } else if (type == RewardType::none) {
    //     reward = high_bw ? knob::hdcpf_reward_hbw_none : knob::hdcpf_reward_none;
    // } else if (type == RewardType::out_of_bounds) {
    //     reward = high_bw ? knob::hdcpf_reward_hbw_out_of_bounds : knob::hdcpf_reward_out_of_bounds;
    // } else if (type == RewardType::tracker_hit) {
    //     reward = high_bw ? knob::hdcpf_reward_hbw_tracker_hit : knob::hdcpf_reward_tracker_hit;
    // } else {
    //     cout << "Invalid reward type found " << type << endl;
    //     assert(false);
    // }

    return reward;
}

void HDCPF::train(HDCPF_PTEntry* curr_evicted, HDCPF_PTEntry* last_evicted) {
    MYLOG("victim %s %u %d last_victim %s %u %d", curr_evicted->state->to_string().c_str(), curr_evicted->action_index, Actions[curr_evicted->action_index],
          last_evicted->state->to_string().c_str(), last_evicted->action_index, Actions[last_evicted->action_index]);

    stats.train.called++;
    if ((last_evicted->reward_type == RewardType::out_of_bounds) || (last_evicted->reward_type == RewardType::tracker_hit)) {
        if (last_evicted->true_action_valid) {
        } else {
            // 这里要更新一下stentry,让准备下一次访问到来的时候,学习一下;

            uint64_t page = last_evicted->page_trigger;
            HDCPF_STEntry* stentry = NULL;
            auto st_index = find_if(signature_table.begin(), signature_table.end(), [page](HDCPF_STEntry* stentry) { return stentry->page == page; });
            if (st_index != signature_table.end()) {
                // (*st_index)->track_prefetch(pred_offset, pref_offset);
                (*st_index)->learnByNextDelta = true;
                (*st_index)->eState.h_pc = last_evicted->encodedState->h_pc;
                (*st_index)->eState.h_seq = last_evicted->encodedState->h_seq;
                (*st_index)->eState.combined = last_evicted->encodedState->combined;
            }
        }
    }
    if (!last_evicted->has_reward) {
        stats.train.compute_reward++;
        reward(last_evicted);
    }
    assert(last_evicted->has_reward);

    /* train */
    brain_hdc_basic->updateModel((last_evicted->encodedState), last_evicted->action_index, last_evicted->reward);
    // if ((last_evicted->reward < 0) && last_evicted->true_action_valid && (last_evicted->action_index != last_evicted->true_action)) {
    //     brain_hdc_basic->updateModel_true((last_evicted->encodedState), last_evicted->true_action, 1);
    // }

    MYLOG("train done");
}

/* TODO: what if multiple prefetch request generated the same address?
 * Currently it just sets the fill bit of the oldest prefetch request.
 * Do we need to set it for everyone? */
void HDCPF::register_fill(uint64_t address) {
    MYLOG("fill @ %lx", address);

    stats.register_fill.called++;
    vector<HDCPF_PTEntry*> ptentries = search_pt(address, knob::hdcpf_enable_reward_all);
    if (!ptentries.empty()) {
        stats.register_fill.set++;
        for (uint32_t index = 0; index < ptentries.size(); ++index) {
            stats.register_fill.set_total++;
            ptentries[index]->is_filled = true;
            MYLOG("fill PT hit. pref with act_idx %u act %d", ptentries[index]->action_index, Actions[ptentries[index]->action_index]);
        }
    }
}

void HDCPF::register_prefetch_hit(uint64_t address) {
    MYLOG("pref_hit @ %lx", address);

    stats.register_prefetch_hit.called++;
    vector<HDCPF_PTEntry*> ptentries = search_pt(address, knob::hdcpf_enable_reward_all);
    if (!ptentries.empty()) {
        stats.register_prefetch_hit.set++;
        for (uint32_t index = 0; index < ptentries.size(); ++index) {
            stats.register_prefetch_hit.set_total++;
            ptentries[index]->pf_cache_hit = true;
            MYLOG("pref_hit PT hit. pref with act_idx %u act %d", ptentries[index]->action_index, Actions[ptentries[index]->action_index]);
        }
    }
}

vector<HDCPF_PTEntry*> HDCPF::search_pt(uint64_t address, bool search_all) {
    vector<HDCPF_PTEntry*> entries;
    for (uint32_t index = 0; index < prefetch_tracker.size(); ++index) {
        if (prefetch_tracker[index]->address == address) {
            entries.push_back(prefetch_tracker[index]);
            if (!search_all) break;
        }
    }
    return entries;
}
// cyh edit
vector<HDCPF_PTEntry*> HDCPF::search_pt_by_page(uint64_t address, bool search_all) {
    vector<HDCPF_PTEntry*> entries;
    uint64_t page = address >> LOG2_PAGE_SIZE;
    for (uint32_t index = 0; index < prefetch_tracker.size(); ++index) {
        if (prefetch_tracker[index]->page_trigger == page) {
            entries.push_back(prefetch_tracker[index]);
            if (!search_all) break;
        }
    }
    return entries;
}

void HDCPF::update_stats(uint32_t state, uint32_t action_index, uint32_t pref_degree) {
    auto it = state_action_dist.find(state);
    if (it != state_action_dist.end()) {
        it->second[action_index]++;
    } else {
        vector<uint64_t> act_dist;
        act_dist.resize(knob::hdcpf_max_actions, 0);
        act_dist[action_index]++;
        state_action_dist.insert(std::pair<uint32_t, vector<uint64_t>>(state, act_dist));
    }
}

void HDCPF::update_stats(State* state, uint32_t action_index, uint32_t degree) {
    string state_str = state->to_string();
    auto it = state_action_dist2.find(state_str);
    if (it != state_action_dist2.end()) {
        it->second[action_index]++;
        it->second[knob::hdcpf_max_actions]++; /* counts total occurences of this state */
    } else {
        vector<uint64_t> act_dist;
        act_dist.resize(knob::hdcpf_max_actions + 1, 0);
        act_dist[action_index]++;
        act_dist[knob::hdcpf_max_actions]++; /* counts total occurences of this state */
        state_action_dist2.insert(std::pair<string, vector<uint64_t>>(state_str, act_dist));
    }

    auto it2 = action_deg_dist.find(getAction(action_index));
    if (it2 != action_deg_dist.end()) {
        it2->second[degree]++;
    } else {
        vector<uint64_t> deg_dist;
        deg_dist.resize(MAX_HDCPF_DEGREE, 0);
        deg_dist[degree]++;
        action_deg_dist.insert(std::pair<int32_t, vector<uint64_t>>(getAction(action_index), deg_dist));
    }
}

int32_t HDCPF::getAction(uint32_t action_index) {
    assert(action_index < Actions.size());
    return Actions[action_index];
}

void HDCPF::track_in_st(uint64_t page, uint32_t pred_offset, int32_t pref_offset) {
    auto st_index = find_if(signature_table.begin(), signature_table.end(), [page](HDCPF_STEntry* stentry) { return stentry->page == page; });
    if (st_index != signature_table.end()) {
        (*st_index)->track_prefetch(pred_offset, pref_offset);
    }
}

void HDCPF::update_bw(uint8_t bw) {
    assert(bw < DRAM_BW_LEVELS);
    bw_level = bw;
    stats.bandwidth.epochs++;
    stats.bandwidth.histogram[bw_level]++;
}

void HDCPF::update_ipc(uint8_t ipc) {
    assert(ipc < HDCPF_MAX_IPC_LEVEL);
    core_ipc = ipc;
    stats.ipc.epochs++;
    stats.ipc.histogram[ipc]++;
}

void HDCPF::update_acc(uint32_t acc) {
    assert(acc < CACHE_ACC_LEVELS);
    acc_level = acc;
    stats.cache_acc.epochs++;
    stats.cache_acc.histogram[acc]++;
}

bool HDCPF::is_high_bw() {
    return bw_level >= knob::hdcpf_high_bw_thresh ? true : false;
}

void HDCPF::dump_stats() {
    cout << "hdcpf_st_lookup " << stats.st.lookup << endl
         << "hdcpf_st_hit " << stats.st.hit << endl
         << "hdcpf_st_evict " << stats.st.evict << endl
         << "hdcpf_st_insert " << stats.st.insert << endl
         << "hdcpf_st_streaming " << stats.st.streaming << endl
         << endl

         << "hdcpf_predict_called " << stats.predict.called << endl
         // << "hdcpf_predict_shaggy_called " << stats.predict.shaggy_called << endl
         << "hdcpf_predict_out_of_bounds " << stats.predict.out_of_bounds << endl;

    for (uint32_t index = 0; index < Actions.size(); ++index) {
        cout << "hdcpf_predict_action_" << Actions[index] << " " << stats.predict.action_dist[index] << endl;
        cout << "hdcpf_predict_issue_action_" << Actions[index] << " " << stats.predict.issue_dist[index] << endl;
        cout << "hdcpf_predict_hit_action_" << Actions[index] << " " << stats.predict.pred_hit[index] << endl;
        cout << "hdcpf_predict_out_of_bounds_action_" << Actions[index] << " " << stats.predict.out_of_bounds_dist[index] << endl;
    }

    cout << "hdcpf_predict_multi_deg_called " << stats.predict.multi_deg_called << endl
         << "hdcpf_predict_predicted " << stats.predict.predicted << endl
         << "hdcpf_predict_multi_deg " << stats.predict.multi_deg << endl;
    for (uint32_t index = 2; index <= MAX_HDCPF_DEGREE; ++index) {
        cout << "hdcpf_predict_multi_deg_" << index << " " << stats.predict.multi_deg_histogram[index] << endl;
    }
    cout << endl;
    for (uint32_t index = 1; index <= MAX_HDCPF_DEGREE; ++index) {
        cout << "hdcpf_selected_deg_" << index << " " << stats.predict.deg_histogram[index] << endl;
    }
    cout << endl;

    if (knob::hdcpf_enable_state_action_stats) {
        if (knob::hdcpf_enable_featurewise_engine) {
            std::vector<std::pair<string, vector<uint64_t>>> pairs;
            for (auto itr = state_action_dist2.begin(); itr != state_action_dist2.end(); ++itr)
                pairs.push_back(*itr);
            sort(pairs.begin(), pairs.end(), [](std::pair<string, vector<uint64_t>>& a, std::pair<string, vector<uint64_t>>& b) { return a.second[knob::hdcpf_max_actions] > b.second[knob::hdcpf_max_actions]; });
            for (auto it = pairs.begin(); it != pairs.end(); ++it) {
                cout << "hdcpf_state_" << hex << it->first << dec << " ";
                for (uint32_t index = 0; index < it->second.size(); ++index) {
                    cout << it->second[index] << ",";
                }
                cout << endl;
            }
        } else {
            for (auto it = state_action_dist.begin(); it != state_action_dist.end(); ++it) {
                cout << "hdcpf_state_" << hex << it->first << dec << " ";
                for (uint32_t index = 0; index < it->second.size(); ++index) {
                    cout << it->second[index] << ",";
                }
                cout << endl;
            }
        }
    }
    cout << endl;

    for (auto it = action_deg_dist.begin(); it != action_deg_dist.end(); ++it) {
        cout << "hdcpf_action_" << it->first << "_deg_dist ";
        for (uint32_t index = 0; index < MAX_HDCPF_DEGREE; ++index) {
            cout << it->second[index] << ",";
        }
        cout << endl;
    }
    cout << endl;

    cout << "hdcpf_track_called " << stats.track.called << endl
         << "hdcpf_track_same_address " << stats.track.same_address << endl
         << "hdcpf_track_evict " << stats.track.evict << endl
         << endl

         << "hdcpf_reward_demand_called " << stats.reward.demand.called << endl
         << "hdcpf_reward_demand_pt_not_found " << stats.reward.demand.pt_not_found << endl
         << "hdcpf_reward_demand_pt_found " << stats.reward.demand.pt_found << endl
         << "hdcpf_reward_demand_pt_found_total " << stats.reward.demand.pt_found_total << endl
         << "hdcpf_reward_demand_has_reward " << stats.reward.demand.has_reward << endl
         << "hdcpf_reward_train_called " << stats.reward.train.called << endl
         << "hdcpf_reward_assign_reward_called " << stats.reward.assign_reward.called << endl
         << "hdcpf_reward_no_pref " << stats.reward.no_pref << endl
         << "hdcpf_reward_incorrect " << stats.reward.incorrect << endl
         << "hdcpf_reward_correct_untimely " << stats.reward.correct_untimely << endl
         << "hdcpf_reward_correct_timely " << stats.reward.correct_timely << endl
         << "hdcpf_reward_out_of_bounds " << stats.reward.out_of_bounds << endl
         << "hdcpf_reward_tracker_hit " << stats.reward.tracker_hit << endl
         << endl;

    for (uint32_t reward = 0; reward < RewardType::num_rewards; ++reward) {
        cout << "hdcpf_reward_" << getRewardTypeString((RewardType)reward) << "_low_bw " << stats.reward.compute_reward.dist[reward][0] << endl
             << "hdcpf_reward_" << getRewardTypeString((RewardType)reward) << "_high_bw " << stats.reward.compute_reward.dist[reward][1] << endl;
    }
    cout << endl;

    for (uint32_t action = 0; action < Actions.size(); ++action) {
        cout << "hdcpf_reward_" << Actions[action] << " ";
        for (uint32_t reward = 0; reward < RewardType::num_rewards; ++reward) {
            cout << stats.reward.dist[action][reward] << ",";
        }
        cout << endl;
    }

    cout << endl
         << "hdcpf_train_called " << stats.train.called << endl
         << "hdcpf_train_compute_reward " << stats.train.compute_reward << endl
         << endl

         << "hdcpf_register_fill_called " << stats.register_fill.called << endl
         << "hdcpf_register_fill_set " << stats.register_fill.set << endl
         << "hdcpf_register_fill_set_total " << stats.register_fill.set_total << endl
         << endl

         << "hdcpf_register_prefetch_hit_called " << stats.register_prefetch_hit.called << endl
         << "hdcpf_register_prefetch_hit_set " << stats.register_prefetch_hit.set << endl
         << "hdcpf_register_prefetch_hit_set_total " << stats.register_prefetch_hit.set_total << endl
         << endl

         << "hdcpf_pref_issue_hdcpf " << stats.pref_issue.hdcpf << endl
         // << "hdcpf_pref_issue_shaggy " << stats.pref_issue.shaggy << endl
         << endl;

    std::vector<std::pair<string, uint64_t>> pairs;
    for (auto itr = target_action_state.begin(); itr != target_action_state.end(); ++itr)
        pairs.push_back(*itr);
    sort(pairs.begin(), pairs.end(), [](std::pair<string, uint64_t>& a, std::pair<string, uint64_t>& b) { return a.second > b.second; });
    for (auto it = pairs.begin(); it != pairs.end(); ++it) {
        cout << it->first << "," << it->second << endl;
    }

    if (brain_hdc_basic) {
        brain_hdc_basic->dump_stats();
    }

    recorder->dump_stats();

    cout << "hdcpf_bw_epochs " << stats.bandwidth.epochs << endl;
    for (uint32_t index = 0; index < DRAM_BW_LEVELS; ++index) {
        cout << "hdcpf_bw_level_" << index << " " << stats.bandwidth.histogram[index] << endl;
    }
    cout << endl;

    cout << "hdcpf_ipc_epochs " << stats.ipc.epochs << endl;
    for (uint32_t index = 0; index < HDCPF_MAX_IPC_LEVEL; ++index) {
        cout << "hdcpf_ipc_level_" << index << " " << stats.ipc.histogram[index] << endl;
    }
    cout << endl;

    cout << "hdcpf_cache_acc_epochs " << stats.cache_acc.epochs << endl;
    for (uint32_t index = 0; index < CACHE_ACC_LEVELS; ++index) {
        cout << "hdcpf_cache_acc_level_" << index << " " << stats.cache_acc.histogram[index] << endl;
    }
    cout << endl;
}
