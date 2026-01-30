#ifndef HDC_PF_HELPER_H
#define HDC_PF_HELPER_H

#include <deque>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "bitmap.h"
#include "champsim.h"
#include "hyperstream/core/hypervector.hpp"
#include "learning_engine_hdc.h"
using namespace std;
// 包含输入,输出,统计类型

#define MAX_HOP_COUNT 16
// 编码结果保存
struct EncodedState {
    BinaryHV h_pc;
    BinaryHV h_seq;
    BinaryHV combined;  // 如果需要 bundle(h_pc, h_seq)
};

// 特征类型枚举
typedef enum {
    PC = 0,
    Offset,
    Delta,
    PC_path,
    Offset_path,
    Delta_path,
    Address,
    Page,

    NumFeatures
} Feature;

const char* getFeatureString(Feature feature);

// 奖励类型枚举
typedef enum {
    none = 0,          // 无预取或未知状态
    incorrect,         // 错误,预取到了,但是直到被驱逐都未被访问
    correct_untimely,  // 正确但不及时
    correct_timely,    // 正确且及时
    out_of_bounds,     // 越出物理页面边界
    tracker_hit,       // tracker命中,想预取,但是发现已经在tracker里了(重复了)
    notsure,           // 被驱逐前还没有use

    num_rewards
} RewardType;

const char* getRewardTypeString(RewardType type);

// 辅助函数:判断奖励是正/反面反馈
inline bool isRewardCorrect(RewardType type) { return (type == correct_timely || type == correct_untimely); }
inline bool isRewardIncorrect(RewardType type) { return type == incorrect; }

// State 类 : 算法在决策时刻时候看到的处理器"快照"
class State {
   public:
    uint64_t pc;
    uint64_t address;
    uint64_t page;
    uint32_t offset;
    int32_t delta;
    uint32_t local_delta_sig;
    uint32_t local_delta_sig2;
    uint32_t local_pc_sig;
    uint32_t local_offset_sig;
    uint8_t bw_level;
    bool is_high_bw;
    uint32_t acc_level;

    /*
     * Add more states here
     */

    void reset() {
        pc = 0xdeadbeef;
        address = 0xdeadbeef;
        page = 0xdeadbeef;
        offset = 0;
        delta = 0;
        local_delta_sig = 0;
        local_delta_sig2 = 0;
        local_pc_sig = 0;
        local_offset_sig = 0;
        bw_level = 0;
        is_high_bw = false;
        acc_level = 0;
    }
    State() { reset(); }
    ~State() {}
    // 压缩所有特征=>单一的hash值
    uint32_t value(); /* apply as many state types as you want */
    // 辅助hash函数
    uint32_t get_hash(uint64_t value); /* play wild with hashes */
    // 调试打印函数
    std::string to_string();
};

// 统计在某个页面上某个动作的置信度
class ActionTracker {
   public:
    int32_t action;
    int32_t conf;
    ActionTracker(int32_t act, int32_t c) : action(act), conf(c) {}
    ~ActionTracker() {}
};

// 物理页历史访问记录 => 每一个对象对应一个物理页(4KB page)
class HDCPF_STEntry {
   public:
    uint64_t page;
    deque<uint64_t> pcs;      // pc列表
    deque<uint32_t> offsets;  // offset列表
    deque<int32_t> deltas;    // delta列表
    // 位图
    Bitmap bmp_real;
    Bitmap bmp_pred;
    // 集合,去重统计
    unordered_set<uint64_t> unique_pcs;
    unordered_set<int32_t> unique_deltas;
    // 首次触发该页时的相关状态信息
    uint64_t trigger_pc;
    uint32_t trigger_offset;
    // 是否呈现流式访问模式
    bool streaming;

    /* tracks last n actions on a page to determine degree */
    // 用于调整 动态预取度
    deque<ActionTracker*> action_tracker;
    unordered_set<int32_t> action_with_max_degree;
    unordered_set<int32_t> afterburning_actions;

    // 该页的总计预取次数
    uint32_t total_prefetches;

    /**
     * 学习 :
     * 如果被驱逐之前都没有反馈的话,就由stentry来辅助学习
     * 要有编码状态,学习有效位,
     */
    EncodedState eState;
    bool learnByNextDelta;

   public:
    // 构造&析构函数
    HDCPF_STEntry(uint64_t p, uint64_t pc, uint32_t offset) : page(p) {
        pcs.clear();
        offsets.clear();
        deltas.clear();
        bmp_real.reset();
        unique_pcs.clear();
        unique_deltas.clear();
        trigger_pc = pc;
        trigger_offset = offset;
        streaming = false;

        pcs.push_back(pc);
        offsets.push_back(offset);
        unique_pcs.insert(pc);
        bmp_real[offset] = 1;

        learnByNextDelta = false;
    }
    ~HDCPF_STEntry() {}
    // 特征计算
    uint32_t get_delta_sig();
    uint32_t get_delta_sig2();
    uint32_t get_pc_sig();
    uint32_t get_offset_sig();
    // 有新访问时调用,更新pc/offset/delta序列
    void update(uint64_t page, uint64_t pc, uint32_t offset, uint64_t address);
    // 记录一次预取行为(更新位图)
    void track_prefetch(uint32_t offset, int32_t pref_offset);
    // 维护 ActionTracker
    void insert_action_tracker(int32_t pref_offset);
    bool search_action_tracker(int32_t action, int32_t& conf);
};

// 预取条目(评估队列),记录一次预取请求
class HDCPF_PTEntry {
   public:
    uint64_t address;
    uint64_t address_trigger;  // 触发这次预取的地址
    uint64_t page_trigger;
    uint64_t offset_trigger;
    uint32_t true_action;
    bool true_action_valid;
    State* state;
    uint32_t action_index;
    /* set when prefetched line is filled into cache
     * check during reward to measure timeliness */
    bool is_filled;
    /* set when prefetched line is alredy found in cache
     * donotes extreme untimely prefetch */
    // 预取已经存在于cache/mshr了=>意味着预取太晚/预取重复
    bool pf_cache_hit;
    // reward反馈
    int32_t reward;
    RewardType reward_type;
    // 是否已经计算过reward
    bool has_reward;
    // 记录哪些特征的featurewise决定选择了这条预取条目的action
    vector<bool> consensus_vec;  // only used in featurewise engine
    EncodedState* encodedState;

    // 构造&析构
    HDCPF_PTEntry(uint64_t ad, uint64_t adt, State* st, uint32_t ac, EncodedState* encodedState) : address(ad), address_trigger(adt), state(st), action_index(ac), encodedState(encodedState) {
        page_trigger = address_trigger >> LOG2_PAGE_SIZE;
        offset_trigger = (address_trigger >> LOG2_BLOCK_SIZE) & ((1ull << (LOG2_PAGE_SIZE - LOG2_BLOCK_SIZE)) - 1);
        is_filled = false;
        pf_cache_hit = false;
        reward = 0;
        reward_type = RewardType::none;
        has_reward = false;

        true_action = 0;
        true_action_valid = false;
    }
    ~HDCPF_PTEntry() {}
};

/* some data structures to mine information from workloads */
// 统计与调试
class HDCPFRecorder {
   public:
    unordered_set<uint64_t> unique_pcs;
    unordered_set<uint64_t> unique_trigger_pcs;
    unordered_set<uint64_t> unique_pages;
    // 统计访问位图的分布
    unordered_map<uint64_t, uint64_t> access_bitmap_dist;
    // vector<unordered_map<int32_t, uint64_t>> hop_delta_dist;
    /* 不同跳数下的delta分布 => 跳数是指访问序列的距离,hop=1指紧邻的两次访问 */
    uint64_t hop_delta_dist[MAX_HOP_COUNT + 1][127];
    uint64_t total_bitmaps_seen;
    uint64_t unique_bitmaps_seen;
    // 不同pc在不同带宽下出现的频率
    unordered_map<uint64_t, vector<uint64_t> > pc_bw_dist;

    HDCPFRecorder() {}
    ~HDCPFRecorder() {}
    // record函数,每次访问到来时调用
    void record_access(uint64_t pc, uint64_t address, uint64_t page, uint32_t offset, uint8_t bw_level);
    void record_trigger_access(uint64_t page, uint64_t pc, uint32_t offset);
    void record_access_knowledge(HDCPF_STEntry* stentry);
    // 在模拟结束时打印统计结果
    void dump_stats();
};

/* 辅助打印函数 */
void print_access_debug(HDCPF_STEntry* stentry);
string print_active_features(vector<int32_t> active_features);
string print_active_features2(vector<int32_t> active_features);

#endif /* HDC_PF_HELPER_H */
