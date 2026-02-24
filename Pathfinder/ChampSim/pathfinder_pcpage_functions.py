# created on Oct 11

import sys
sys.path.insert(0, './bindsnet')

import torch

from bindsnet.models import DiehlAndCook2015
from bindsnet.network.monitors import Monitor
from bindsnet import encoding
from collections import OrderedDict
import numpy as np
import os
import time

torch.set_num_threads(1)
torch.set_num_interop_threads(1)
# print("torch num threads:", torch.get_num_threads())
# print("interop threads:", torch.get_num_interop_threads())
# print("cpu count:", os.cpu_count())

right_reward = 1
wrong_reward = -1
conf_threshold = 4
conf_max = 12 # 5
conf_min = -3 # -2
max_actions_per_neuron = 5
class CreateNetwork:

    def __init__(self, pattern_length, confidence_threshold, min_confidence, delta_range_length, neuron_numbers, timestamps, input_intensity,debug_amplitude):
        self.prediction_table = {}  # key = neuron, value = tuple(label, confidence)

        self.pattern_length = pattern_length
        self.confidence_threshold = confidence_threshold
        self.min_confidence = min_confidence
        self.delta_range_length = delta_range_length
        self.neuron_numbers = neuron_numbers
        self.timestamps = timestamps
        self.input_intensity = input_intensity

        self.amplitude = 32/self.timestamps
        self.total_inputs = self.delta_range_length * self.pattern_length
        self.debug_amplitude = debug_amplitude

        self.network = DiehlAndCook2015(
            n_inpt=self.delta_range_length * self.pattern_length,
            n_neurons=self.neuron_numbers,
            exc=25.5,
            inh=20.5,
            # inh=4.5,
            # exc=25,
            # inh=6.5,
            dt=1,
            norm=(self.delta_range_length*self.pattern_length)/10,
            theta_plus=.05,
            inpt_shape=(1, self.delta_range_length * self.pattern_length),
        )

        # Simulation time.
        time = self.timestamps
        dt = 1
        device = 'cpu'

        # set up the spike monitors
        self.spikes = {}
        self.voltages = {}
        for layer in set(self.network.layers):
            self.spikes[layer] = Monitor(
                self.network.layers[layer], state_vars=["s"], time=int(time / dt), device=device
            )
            self.network.add_monitor(self.spikes[layer], name="%s_spikes" % layer)


        self.output_monitor = self.spikes['Ae']
        self.voltages['Ae'] = Monitor(
            self.network.layers['Ae'], state_vars=["v"], time=int(time / dt), device=device
        )
        self.network.add_monitor(self.voltages['Ae'], name="%s_voltage" % layer)

    def feed_input(self, delta_pattern):
        output_neurons = []
        # s_in = input_data_new
        # print("input spikes:", int(s_in.sum()))

        # s_ae = self.output_monitor.get("s")
        # print("Ae spikes:", int(s_ae.sum()))

        # v_ae = self.voltages["Ae"].get("v")
        # print("Ae vmax:", float(v_ae.max()))
        # w = self.network.connections[("X","Ae")].w
        # print("X->Ae w min/max:", float(w.min()), float(w.max()))

        # # 2) 看 Ae 的阈值（字段名可能是 thresh / theta / v_thresh 之类）
        # ae = self.network.layers["Ae"]
        # for name in ["thresh", "theta", "v_thresh"]:
        #     if hasattr(ae, name):
        #         val = getattr(ae, name)
        #         print("Ae", name, float(val.mean()) if hasattr(val, "mean") else val)

        # array for keeping track of firing neurons
        if self.timestamps == 1:
            # time1 = time.perf_counter()
            non_zero_indices = self.build_enlarged_input_array(delta_pattern)
            input_data_new = torch.zeros(1, 1, self.total_inputs, device='cpu')
            input_data_new[0, 0, non_zero_indices] = self.debug_amplitude if self.debug_amplitude is not None else self.amplitude
            test_inputs = {"X": input_data_new}
            # time2 = time.perf_counter()
            self.network.run(inputs=test_inputs, time=self.timestamps,one_step=(self.timestamps==1))
            # time3 = time.perf_counter()
            voltage_record = self.voltages['Ae'].get('v') # 形状: [time, batch, neurons]
            final_voltages = voltage_record[-1, 0, :]
            best_neuron = torch.argmax(final_voltages).item()
            output_neurons = [best_neuron]
            self.network.reset_state_variables()
            # time4 = time.perf_counter()
            # print("time2-time1:",time2-time1)
            # print("time3-time2:",time3-time2)
            # print("time4-time3:",time4-time3)
            return output_neurons
        else:
            current = torch.full([self.delta_range_length * self.pattern_length], 0)
            non_zero_indices = self.build_enlarged_input_array(delta_pattern)
            for index in non_zero_indices:
                current[index] = self.input_intensity
            input_data_new = encoding.encodings.poisson(current, self.timestamps, 1, device='cpu')
            test_inputs = {"X": input_data_new}
            # print(input_data_new)
            self.network.run(inputs=test_inputs, time=self.timestamps,one_step=(self.timestamps==1))

            # spikes = self.output_monitor.get('s')[:, 0, :]
            # spike_indices = torch.nonzero(spikes)
            # if len(spike_indices) > 0:
            #     first_fired_neuron = spike_indices[0][1].item()
            #     if first_fired_neuron not in output_neurons:
            #         output_neurons.append(first_fired_neuron)
            #     self.network.reset_state_variables()
            #     return output_neurons

            for times in range(self.timestamps):
                for i in range(self.neuron_numbers):
                    if self.output_monitor.get('s')[times][0][i]:
                        # todo break the loops when we get enough output neurons
                        # spikecount[i] += 1
                        # it may have two same neurons fired in certain amount time
                        if i not in output_neurons:
                            output_neurons.append(i)
                        self.network.reset_state_variables()  # Reset state variables.
                        return output_neurons
            self.network.reset_state_variables()  # Reset state variables.
            # print("output_neurons:",output_neurons)
            return output_neurons

    # make the valid information more rich, make the valid pixel thicker
    def build_enlarged_input_array(self, delta_pattern):
        valid_index_list_1d = []
        for i in range(self.pattern_length):

            # TODO reordered input 0 1 2 3 4 5 > 4 1 5 3 0 2
            # version 11: use if, elif to reorder
            # if delta_pattern[i] % 6 == 0 and delta_pattern[i] + 4 <= 63:
            #     valid_index_1d = int(delta_pattern[i] + 4 + (self.delta_range_length - 1) / 2)
            # elif delta_pattern[i] % 6 == 2 and delta_pattern[i] + 3 <= 63:
            #     valid_index_1d = int(delta_pattern[i] + 3 + (self.delta_range_length - 1) / 2)
            # else:
                # version 10 code (remove indent to get back )
            valid_index_1d = int(delta_pattern[i] + (self.delta_range_length - 1) / 2)
            # print(valid_index_1d)
            valid_index_list_1d.append(valid_index_1d)

        input_array_prep = np.empty((0, 127), int)
        for valid_item_index in valid_index_list_1d:
            temp_list = np.zeros(127)
            temp_list = temp_list.astype(int)
            temp_list[valid_item_index] = 1
            input_array_prep = np.vstack((input_array_prep, temp_list))
        valid_index_list_2d = list(zip(*np.where(input_array_prep == 1)))

        row_bound = self.pattern_length - 1
        column_bound = self.delta_range_length - 1

        for index_2d in valid_index_list_2d:
            row_index = index_2d[0]
            column_index = index_2d[1]
            if row_index == row_bound:
                new_valid_row_index = row_index - 1
            else:
                new_valid_row_index = row_index + 1

                # prep_list[index_2d[0]-1, index_2d[1]] = 1
            if column_index == column_bound:
                new_valid_column_index = column_index - 1
            else:
                new_valid_column_index = column_index + 1

            input_array_prep[row_index, new_valid_column_index] = 1
            input_array_prep[new_valid_row_index, column_index] = 1
            input_array_prep[new_valid_row_index, new_valid_column_index] = 1

        input_array_prep = input_array_prep.flatten()
        non_zero_indices = np.where(input_array_prep == 1)[0]

        return non_zero_indices

    def make_prediction(self, delta_pattern):
        output_neurons = self.feed_input(delta_pattern)
        # tuple array for outputNeuron and delta
        prediction_deltas = []
        if output_neurons and output_neurons[0] in self.prediction_table:
            # for delta_tuple in self.prediction_table[output_neurons[0]]:
            # 修改这里：先按照置信度(x[1])降序排序
            sorted_tuples = sorted(self.prediction_table[output_neurons[0]], key=lambda x: x[1], reverse=True)
            for delta_tuple in sorted_tuples:
                # prediction_deltas.append( delta_tuple[0]) 
                prediction_deltas.append(delta_tuple) #把confidence也记录下来了
        else:
            # todo add nextline prefetcher here
            # prediction_delta += 1
            prediction_deltas = None


        # check prediction table if that's okay to generate prediction(if there is neuron label pair )
        # if not add neuron to training table
        # update training table
        # update prediction table, if there is existed neuron # in prediction table
        return output_neurons, prediction_deltas

    def label_unlabeled(self, page, offset, training_table):
        # 直接检查 page 是否在表中
        if page in training_table:
            fired_neuron = training_table[page][2]
            delta = offset - training_table[page][1]
            if fired_neuron >= 0 and fired_neuron not in self.prediction_table:
                # assign new label and confidence
                self.prediction_table[fired_neuron] = [(delta, 0)]
            # add the second label confidence pair
            elif fired_neuron >= 0 and fired_neuron in self.prediction_table:
                # first_delta_tuple = self.prediction_table[fired_neuron][0]
                # if len(self.prediction_table[fired_neuron]) == 1 and delta != first_delta_tuple[0]:
                #     self.prediction_table[fired_neuron].append((delta, 0))
                existing_deltas = [item[0] for item in self.prediction_table[fired_neuron]]
                    # 只有当这个 delta 还没有被记录过时，才考虑添加
                if delta not in existing_deltas:
                    # 检查当前记录的动作个数是否小于我们设置的参数上限
                    if len(self.prediction_table[fired_neuron]) < max_actions_per_neuron:
                        self.prediction_table[fired_neuron].append((delta, 0))


# use the current page to find firing neuron, and use neuron to find the label,
# comparing label with current offset, then update confidence in prediction table
def check_hit(training_table, prediction_table, page, offset, label_removal_counter):
    # 直接检查 page
    if page in training_table:

        fired_neuron = training_table[page][2] 

        if fired_neuron > 0 and fired_neuron in prediction_table:
            is_correct_bit = 0
            # label_toRemove_idx = None
            labels_to_remove_idx = []
            for i in range(0, len(prediction_table[fired_neuron])):
                delta_tuple = prediction_table[fired_neuron][i]
                confidence = delta_tuple[1]
                if delta_tuple[0] == (offset - training_table[page][1]): 
                    if confidence < conf_max:
                        prediction_table[fired_neuron][i] = (delta_tuple[0], confidence + right_reward)
                        is_correct_bit = 1
                    # break

                else:
                    if confidence < conf_min:
                        label_removal_counter += 1
                        labels_to_remove_idx.append(i)
                        # label_toRemove_idx = i
                    else:
                        prediction_table[fired_neuron][i] = (delta_tuple[0], confidence + wrong_reward)
            # if label_toRemove_idx is not None:
                # del prediction_table[fired_neuron][label_toRemove_idx]  
            if labels_to_remove_idx:
                for idx in sorted(labels_to_remove_idx, reverse=True):
                    del prediction_table[fired_neuron][idx]
            return is_correct_bit, label_removal_counter
        else:
            return 0, label_removal_counter
    else:
        return 0, label_removal_counter


def calculate_deltas(inner_dict_values, offset):
    old_deltas = inner_dict_values[0]
    new_deltas = [old_deltas[1], old_deltas[2], (offset-inner_dict_values[1])]

    return new_deltas


def update_training_table(page, offset, fired_neurons, new_delta_pattern, training_table, offset_prediction, page_removal_counter):

    if len(training_table) > 1024:
        training_table.popitem(last=False)
        page_removal_counter += 1

    if fired_neurons:
        training_table[page] = [new_delta_pattern, offset, fired_neurons[0], offset_prediction]
    else:
        training_table[page] = [new_delta_pattern, offset, -1, offset_prediction]
    
    training_table.move_to_end(page)

    return page_removal_counter


