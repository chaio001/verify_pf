#ifndef HDC_PF_H
#define HDC_PF_H

#include <unordered_map>
#include <vector>

#include "cache.h"
#include "champsim.h"
#include "hdc_pf_helper.h"
#include "learning_engine_hdc_basic.h"
#include "prefetcher.h"

using namespace std;

#define MAX_ACTIONS 127         // 预取动作最大数量
#define MAX_REWARDS 16         // 奖励类型最大数量
#define MAX_HDCPF_DEGREE 16    // 预取度最大值
#define HDCPF_MAX_IPC_LEVEL 4  // IPC分级等级数量

/* forward declaration */
class LearningEngine;
extern std::vector<int32_t> Actions;
extern std::unordered_map<int32_t, int32_t> action_to_index_map;
class HDCPF : public Prefetcher {
   private:
    deque<HDCPF_STEntry*> signature_table;  // 记录物理页的访问历史,签名表
    // LearningEngineBasic* brain;                    // 预取引擎_basic
    // LearningEngineFeaturewise* brain_featurewise;  // 预取引擎_featurewise
    LearningEngineHDCBasic* brain_hdc_basic;
    deque<HDCPF_PTEntry*> prefetch_tracker;  // UNSURE : 评估队列 ?
    HDCPF_PTEntry* last_evicted_tracker;     // 用于SARSA更新的,记录上一条目的记录
    uint8_t bw_level;
    uint8_t core_ipc;
    uint32_t acc_level;

    EncodedState* encodedState;  // 当前的编码结果

    HDCPFRecorder* recorder;  // 统计workload特征,暂时用不上

    unordered_map<string, uint64_t> target_action_state;  // 调试专用 (Debugging) , 统计特定的“状态字符串”和“动作”组合出现了多少次

    // 下面的结构体:stats,包含了预取器所有的性能计数器
    struct
    {
        struct
        {
            uint64_t lookup;
            uint64_t hit;
            uint64_t evict;
            uint64_t insert;
            uint64_t streaming;
        } st;  // 签名表的查找次数、命中率、驱逐次数

        struct
        {
            uint64_t called;
            uint64_t out_of_bounds;
            vector<uint64_t> action_dist;
            vector<uint64_t> issue_dist;
            vector<uint64_t> pred_hit;
            vector<uint64_t> out_of_bounds_dist;
            uint64_t predicted;
            uint64_t multi_deg;
            uint64_t multi_deg_called;
            uint64_t multi_deg_histogram[MAX_HDCPF_DEGREE + 1];
            uint64_t deg_histogram[MAX_HDCPF_DEGREE + 1];
        } predict;  // 预取行为统计:预测次数、发出的预取数、越界次数

        struct
        {
            uint64_t called;
            uint64_t same_address;
            uint64_t evict;
        } track;

        struct
        {
            struct
            {
                uint64_t called;
                uint64_t pt_not_found;
                uint64_t pt_found;
                uint64_t pt_found_total;
                uint64_t has_reward;
            } demand;

            struct
            {
                uint64_t called;
            } train;

            struct
            {
                uint64_t called;
            } assign_reward;

            struct
            {
                uint64_t dist[MAX_REWARDS][2];
            } compute_reward;  // 高低带宽下,Reward类型统计数量

            uint64_t correct_timely;
            uint64_t correct_untimely;
            uint64_t no_pref;
            uint64_t incorrect;
            uint64_t out_of_bounds;
            uint64_t tracker_hit;
            uint64_t notsure;
            uint64_t dist[MAX_ACTIONS][MAX_REWARDS];
        } reward;  // 不同reward类型的统计信息

        struct
        {
            uint64_t called;
            uint64_t compute_reward;
        } train;

        struct
        {
            uint64_t called;
            uint64_t set;
            uint64_t set_total;
        } register_fill;

        struct
        {
            uint64_t called;
            uint64_t set;
            uint64_t set_total;
        } register_prefetch_hit;

        struct
        {
            uint64_t hdcpf;
        } pref_issue;

        struct
        {
            uint64_t epochs;
            uint64_t histogram[DRAM_BW_LEVELS];
        } bandwidth;

        struct
        {
            uint64_t epochs;
            uint64_t histogram[HDCPF_MAX_IPC_LEVEL];
        } ipc;

        struct
        {
            uint64_t epochs;
            uint64_t histogram[CACHE_ACC_LEVELS];
        } cache_acc;
    } stats;

    // 更细粒度的状态-动作分布统计,某个状态下,某个动作被选中的次数
    unordered_map<uint32_t, vector<uint64_t> > state_action_dist;
    unordered_map<std::string, vector<uint64_t> > state_action_dist2;
    // 某个动作下预取度的分布
    unordered_map<int32_t, vector<uint64_t> > action_deg_dist;

   private:
    // 读取.ini或者命令行参数,初始化全局配置(Knobs)
    void init_knobs();
    // 初始化stats结构体
    void init_stats();

    // 暂时不考虑
    void update_global_state(uint64_t pc, uint64_t page, uint32_t offset, uint64_t address);
    // 维护局部历史,即维护signature_table数据
    HDCPF_STEntry* update_local_state(uint64_t pc, uint64_t page, uint32_t offset, uint64_t address);
    // 预取决策函数
    uint32_t predict(uint64_t address, uint64_t page, uint32_t offset, State* state, vector<uint64_t>& pref_addr,EncodedState* encodedState);
    // 核心track函数,创建预取记录对象PTEntry,放到prefetch_tracker中,如果队列满了->驱逐->触发train()
    bool track(uint64_t address,uint64_t address_trigger, State* state, uint32_t action_index, HDCPF_PTEntry** tracker, EncodedState* encodedState);
    // 按需奖励(对比访存地址和之前的预取地址,然后计算reward)
    void reward(uint64_t address);
    // 驱逐奖励(队里数据被驱逐时,计算reward)
    void reward(HDCPF_PTEntry* ptentry);
    // Setter函数,将reward值写如预取条目中,并更新stats统计数据
    void assign_reward(HDCPF_PTEntry* ptentry, RewardType type);
    // 计算reward的逻辑
    int32_t compute_reward(HDCPF_PTEntry* ptentry, RewardType type);
    // 学习/train函数
    void train(HDCPF_PTEntry* curr_evicted, HDCPF_PTEntry* last_evicted);
    // 基于地址查找prefetch_tracker队列中对应的Entry(条目)
    vector<HDCPF_PTEntry*> search_pt(uint64_t address, bool search_all = false);
    vector<HDCPF_PTEntry*> search_pt_by_page(uint64_t address, bool search_all = false);
    // 更新state_action_dist&state_action_dist2表
    void update_stats(uint32_t state, uint32_t action_index, uint32_t pref_degree = 1);
    void update_stats(State* state, uint32_t action_index, uint32_t degree = 1);
    // 标记"物理页的访问历史"中的预取位图
    void track_in_st(uint64_t page, uint32_t pred_offset, int32_t pref_offset);
    // 根据预取度生成更多的预取地址
    void gen_multi_degree_pref(uint64_t page, uint32_t offset, int32_t action, uint32_t pref_degree, vector<uint64_t>& pref_addr);
    // 基于Q值动态调整Degree,预取度
    uint32_t get_dyn_pref_degree(float max_to_avg_q_ratio, uint64_t page = 0xdeadbeef, int32_t action = 0); /* only implemented for CMAC engine 2.0 */
    // 判断当前内存带宽利用率是否超过阈值
    bool is_high_bw();

   public:
    // 构造&析构函数
    HDCPF(string type);
    ~HDCPF();
    // 循环内执行的入口函数
    void invoke_prefetcher(uint64_t pc, uint64_t address, uint8_t cache_hit, uint8_t type, vector<uint64_t>& pref_addr);
    // 预取数据进入缓存时调用,标记预取条目的is)_filled位
    void register_fill(uint64_t address);
    // 代表了 “冗余预取”,当预取已经存在于cache中,或者正在被预取(存在于MSHR中),就会调用这个函数
    void register_prefetch_hit(uint64_t address);
    // 模拟结束后的打印信息
    void dump_stats();
    void print_config();
    // action_index转换到具体的预取动作,action
    int32_t getAction(uint32_t action_index);
    // 接受来自ChampSim系统的环境统计信息,用于辅助决策
    void update_bw(uint8_t bw_level);
    void update_ipc(uint8_t ipc);
    void update_acc(uint32_t acc_level);
};

#endif /* HDC_PF_H */
