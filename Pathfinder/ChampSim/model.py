import time
from abc import ABC, abstractmethod
import torch
import pathfinder_pcpage_functions as pf_pcpage
from collections import OrderedDict

class MLPrefetchModel_pathfinder_pcpage(object):
    '''
    Abstract base class for your models. For HW-based approaches such as the
    NextLineModel below, you can directly add your prediction code. For ML
    models, you may want to use it as a wrapper, but alternative approaches
    are fine so long as the behavior described below is respected.
    '''
    def __init__(self,debug_amplitude=None):
        self.debug_amplitude = debug_amplitude

    @abstractmethod
    def load(self, path):
        '''
        Loads your model from the filepath path
        '''
        pass

    @abstractmethod
    def save(self, path):
        '''
        Saves your model to the filepath path
        '''
        pass

    @abstractmethod
    def train(self, data):
        '''
        Train your model here. No return value. The data parameter is in the
        same format as the load traces. Namely,
        Unique Instr Id, Cycle Count, Load Address, Instruction Pointer of the Load, LLC hit/miss
        '''
        pass

    @abstractmethod
    def generate(self, data):
        print(torch.cuda.is_available())
        # print(torch.cuda.current_device())
        # print(torch.cuda.get_device_name())
        # print(torch.cuda.get_device_capability())
        print(len(data))
        '''
        Generate your prefetches here. Remember to limit yourself to 2 prefetches
        for each instruction ID and to not look into the future :).

        The return format for this will be a list of tuples containing the
        unique instruction ID and the prefetch. For example,
        [
            (A, A1),
            (A, A2),
            (C, C1),
            ...
        ]

        where A, B, and C are the unique instruction IDs and A1, A2 and C1 are
        the prefetch addresses.
        '''

        # constants
        pattern_length = 3
        confidence_threshold = 1
        min_confidence = -2
        correct_predictions = 0
        correct_offset_predictions = 0
        correct_delta_predictions = 0
        total_predictions = 0
        total_delta_prediction = 0
        total_offset_prediction = 0
        label_removal_counter = 0
        pc_removal_counter = 0
        page_removal_counter = 0

        # constants for snn
        delta_range_length = 127
        neuron_numbers = 50

        # timestamps = 32
        timestamps = 1
        # input_intensity = 50432
        # input_intensity = 10000
        input_intensity = 1576

        sanity_check_index = 3000
        refresh_index = 5000
        last_addr_index = 1000000
        # last_addr_index = 1000000000

        # create snn
        single_network = pf_pcpage.CreateNetwork(pattern_length, confidence_threshold, min_confidence, delta_range_length, neuron_numbers, timestamps, input_intensity,self.debug_amplitude)

        prefetch_addresses = []
        # key = pc, value = dict_inner {key = page, value = [[delta pattern(length = 3)], last offset, neuron, offset_pre/delta_pre]}
        training_table = OrderedDict()

        cur_index = 0

        for (instr_id, cycle_count, load_addr, load_ip, llc_hit) in data:
            # print("---------------------------------")
            # print("current index: ", cur_index)
            # time.sleep(5)

            if cur_index % sanity_check_index == 0 and cur_index != 0:
                print("----- the current index: ", cur_index)
                print("correct prediction: ", correct_predictions)
                print("total predictions: ", total_predictions)
                print("correct/total predict: ", correct_predictions / total_predictions)
                print(" correct / total load instr: ", correct_predictions / cur_index)
                print(" ")
            if cur_index > last_addr_index:
                break
            if cur_index % refresh_index == 0 and cur_index != 0:
                single_network = pf_pcpage.CreateNetwork(pattern_length, confidence_threshold, min_confidence, delta_range_length, neuron_numbers, timestamps, input_intensity,self.debug_amplitude)

            cur_index += 1
            pc, page, offset = load_ip, load_addr >> 12, ((load_addr >> 6) & 0x3f)

            is_correct, label_removal_counter = pf_pcpage.check_hit(training_table, single_network.prediction_table, page, offset, label_removal_counter)
            if is_correct:
                correct_predictions += 1
                if training_table[page][3] == 1:
                    correct_offset_predictions += 1
                else:
                    correct_delta_predictions += 1

            single_network.label_unlabeled(page, offset, training_table)

            # key = pc, value = inner_dict {key = page, value = [[delta pattern(length = 3)], last offset, neuron, offset_pre/delta_pre]}
            offset_prediction = 0
            # if pc in training_table:
            #     inner_dict = training_table[pc]

            if page in training_table:
                # 直接传入 training_table[page] 获取数据
                delta_pattern = pf_pcpage.calculate_deltas(training_table[page], offset)
                offset_prediction = 0 # 或者是从表中获取之前的状态，具体看原有逻辑需求，这里通常设为0表示有历史
            else:
                # 没有历史记录，使用默认模式 (假设 pattern_length=3)
                # 注意：这里 [offset, 0, 0] 是原代码的初始化方式，保持一致即可
                delta_pattern = [offset, 0, 0] 
                # delta_pattern = [0, 0, 0] 
                offset_prediction = 1

            # ----------------------has some problem
            # start = time.perf_counter()
            output_neurons, predicted_deltas = single_network.make_prediction(delta_pattern)
            # end = time.perf_counter()
            # print(f"end - start   部分耗时: {end - start:.6f} 秒")

            if predicted_deltas is not None:
                for predicted_delta, confidence in predicted_deltas:

                    # if confidence < 0:  # <--- 自由调节这个数字！
                    #     continue 

                    # 检查地址合法性
                    # if offset + predicted_delta > 0:
                    target_offset = offset + predicted_delta
                    if 0 <= target_offset < 64:
                        if offset_prediction == 1:
                            total_offset_prediction += 1
                        else:
                            total_delta_prediction += 1

                        total_predictions += 1
                        predicted_address = (int(page) << 12) + (int(offset + predicted_delta) << 6)
                        
                        # 注意：追加到 prefetch_addresses 的依然是 2元组
                        # 这样就不会破坏后续 generate_prefetch_file 写入 .txt 文件的格式
                        prefetch_addresses.append((instr_id, predicted_address, confidence))

            # if predicted_deltas is not None:
            #     # todo why no error before?
            #     for predicted_delta in predicted_deltas:
            #         if offset + predicted_delta > 0:
            #             if offset_prediction == 1:
            #                 total_offset_prediction += 1
            #             else:
            #                 total_delta_prediction += 1

            #             total_predictions += 1
            #             predicted_address = (int(page) << 12) + (int(offset + predicted_delta) << 6)
            #             prefetch_addresses.append((instr_id, predicted_address))

            # if output_neurons:
            page_removal_counter = pf_pcpage.update_training_table(page, offset, output_neurons, delta_pattern, training_table, offset_prediction, page_removal_counter)
            # print("check training table", training_table)

        print("correct prediction: ", correct_predictions)
        print("total predictions: ", total_predictions)
        print("correct/total predict: ", correct_predictions / total_predictions)
        print(" correct / total load instr: ", correct_predictions / cur_index)

        print("correct offset prediction: ", correct_offset_predictions)
        print("total offset prediction: ", total_offset_prediction)
        # print("correct offset prediction/total offset prediction:", correct_offset_predictions/total_offset_prediction)

        print("correct delta prediction: ", correct_delta_predictions)
        print("total delta prediction: ", total_delta_prediction)
        print("correct delta prediction/total delta prediction:", correct_delta_predictions / total_delta_prediction)

        print("# of prefetches ", len(prefetch_addresses))

        return prefetch_addresses
# Replace this if you create your own model
Model = MLPrefetchModel_pathfinder_pcpage