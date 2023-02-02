import pickle as pkl
import os
import pandas as pd
import numpy as np
import json
import copy
from math import isnan
import matplotlib as mlp
import shutil
import argparse
from sklearn.linear_model import LinearRegression

import matplotlib as mlp
#mlp.use("agg")
# mlp.rcParams.update(mlp.rcParamsDefault)
import matplotlib.pyplot as plt
# plt.rcParams['text.usetex'] = True



#font = {'size': 24}
#mlp.rc('font', **font)

NAN_LABEL = -1


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--memo", type=str, default='Hangzhou_4_4')
    # parser.add_argument("--memo", type=str, default='Jinan_3_4')
    # parser.add_argument("--memo", type=str, default='0515_afternoon_Colight_3_3_bi_high_2')
    # parser.add_argument("--memo", type=str, default='SingleInt')
    parser.add_argument("-b", action="store_true",default=False, help="run baseline analysis") ##
    return parser.parse_args()


def get_traffic_volume(file_name, run_cnt):
    scale = run_cnt / 3600  # run_cnt > traffic_time, no exact scale
    if "synthetic" in file_name:
        sta = file_name.rfind("-") + 1
        print(file_name, int(int(file_name[sta:-4]) * scale))
        return int(int(file_name[sta:-4]) * scale)
    elif "cross" in file_name:
        sta = file_name.find("equal_") + len("equal_")
        end = file_name.find(".xml")
        return int(int(file_name[sta:end]) * scale * 4)  # lane_num = 4


def get_metrics(duration_list, queue_length_list, min_duration, min_duration_id, min_queue_length, min_queue_length_id,
                traffic_name, total_summary, mode_name, save_path, num_rounds, min_duration2=None):
    validation_duration_length = 10
    minimum_round = 50 if num_rounds > 49 else 0
    duration_list = np.array(duration_list)
    queue_length_list = np.array(queue_length_list)

    # min_duration, min_duration_id = np.min(duration_list), np.argmin(duration_list)
    # min_queue_length, min_queue_length_id = np.min(queue_length_list), np.argmin(queue_length_list)

    nan_count = len(np.where(duration_list == NAN_LABEL)[0])
    validation_duration = duration_list[-validation_duration_length:]
    final_duration = np.round(np.mean(validation_duration[validation_duration > 0]), decimals=2)
    final_duration_std = np.round(np.std(validation_duration[validation_duration > 0]), decimals=2)

    if nan_count == 0:
        convergence = {1.2: len(duration_list) - 1, 1.1: len(duration_list) - 1}
        for j in range(minimum_round, len(duration_list)):
            for level in [1.2, 1.1]:
                if max(duration_list[j:]) <= level * final_duration:
                    if convergence[level] > j:
                        convergence[level] = j
        conv_12 = convergence[1.2]
        conv_11 = convergence[1.1]
    else:
        conv_12, conv_11 = 0, 0

    # simple plot for each training instance
    f, ax = plt.subplots(1, 1, figsize=(12, 9), dpi=100)
    ax.plot(duration_list, linewidth=2, color='k')
    ax.plot([0, len(duration_list)], [final_duration, final_duration], linewidth=2, color="g")
    ax.plot([conv_12, conv_12], [duration_list[conv_12], duration_list[conv_12] * 3], linewidth=2, color="b")
    ax.plot([conv_11, conv_11], [duration_list[conv_11], duration_list[conv_11] * 3], linewidth=2, color="b")
    ax.plot([0, len(duration_list)], [min_duration, min_duration], linewidth=2, color="r")
    ax.plot([min_duration_id, min_duration_id], [min_duration, min_duration * 3], linewidth=2, color="r")
    ax.set_title(traffic_name + "-" + str(final_duration))
    plt.savefig(save_path + "/" + traffic_name + "-" + mode_name + ".png")
    plt.close()

    inter_num = traffic_name.split('_')[0]
    vol = traffic_name.split('_')[3]
    ratio = traffic_name.split('_')[4]


    total_summary['inter_num'].append(inter_num)
    total_summary['traffic_volume'].append(vol)
    total_summary['ratio'].append(ratio)

    if ".xml" in traffic_name:
        total_summary["traffic"].append(traffic_name.split(".xml"))
    elif ".json" in traffic_name:
        total_summary["traffic"].append(traffic_name.split(".json"))

    total_summary["min_queue_length"].append(min_queue_length)
    total_summary["min_queue_length_round"].append(min_queue_length_id)
    total_summary["min_duration"].append(min_duration)
    total_summary["min_duration_round"].append(min_duration_id)
    total_summary["final_duration"].append(final_duration)
    total_summary["final_duration_std"].append(final_duration_std)
    total_summary["convergence_1.2"].append(conv_12)
    total_summary["convergence_1.1"].append(conv_11)
    total_summary["nan_count"].append(nan_count)
    total_summary["min_duration2"].append(min_duration2)

    return total_summary



def summary_plot(traffic_performance, figure_dir, mode_name, num_rounds, y_label):

    for traffic_name in traffic_performance:
        f, ax = plt.subplots(1, 1, figsize=(12, 4.5), dpi=200)
        for ti in range(len(traffic_performance[traffic_name])):
            if 'Const' in traffic_performance[traffic_name][ti][1][1:]:
                name='$\it{FAirLight}$'
            elif 'Colight' in traffic_performance[traffic_name][ti][1][1:]:
                name='CoLight'
            elif 'DQN' in traffic_performance[traffic_name][ti][1][1:]:
                name='DQN'
            else:
                name='SAC-GNN'
            if name =='$\it{FAirLight}$':
                plt.plot(traffic_performance[traffic_name][ti][0], label=name, linewidth=3)
            else:
                plt.plot(traffic_performance[traffic_name][ti][0],'--', label=name, linewidth=1.3)                
        plt.legend(bbox_to_anchor=(0.5, 1.18), loc='upper center', ncol=4, fancybox=True, fontsize=18)
        plt.ylabel(y_label, fontsize=18)
        plt.xlabel('Episode', fontsize=18)
        plt.savefig(figure_dir + "/" + traffic_name + "-" + mode_name + ".pdf")
        plt.close()

def summary_plot_throughput_phase(traffic_performance, figure_dir, mode_name, num_rounds, y_label):

    for traffic_name in traffic_performance:
        for j in range(4):
            f, ax = plt.subplots(1, 1, figsize=(12, 9), dpi=200)
            for ti in range(len(traffic_performance[traffic_name])):
                if 'Const' in traffic_performance[traffic_name][ti][1][1:]:
                    name='$\it{FAirLight}$'
                elif 'Colight' in traffic_performance[traffic_name][ti][1][1:]:
                    name='CoLight'
                elif 'DQN' in traffic_performance[traffic_name][ti][1][1:]:
                    name='DQN'
                else:
                    name='SAC-GNN'
                data=[r[j] for r in traffic_performance[traffic_name][ti][0]]
                if name =='$\it{FAirLight}$':
                    plt.plot(data, label=name, linewidth=3)
                else:
                    plt.plot(data,'--', label=name, linewidth=1.3)                
            plt.legend(bbox_to_anchor=(0.5, 1.1), loc='upper center', ncol=4, fancybox=True, fontsize=18)
            plt.ylabel(y_label, fontsize=18)
            plt.xlabel('Episode', fontsize=18)
            plt.savefig(figure_dir + "/" + traffic_name + "-" + mode_name+str(j)+".pdf")
            plt.close()

# def summary_plot(traffic_performance, figure_dir, mode_name, num_rounds, y_label):
#     minimum_round = 50 if num_rounds > 50 else 0
#     validation_duration_length = 10
#     anomaly_threshold = 1.3

#     for traffic_name in traffic_performance:
#         f, ax = plt.subplots(2, 1, figsize=(12, 9), dpi=100)
#         performance_tmp = []
#         check_list = []
#         for ti in range(len(traffic_performance[traffic_name])):
#             if 'Const' in traffic_performance[traffic_name][ti][1][1:]:
#                 name='SAC-GNN Constrained'
#             elif 'Colight' in traffic_performance[traffic_name][ti][1][1:]:
#                 name='CoLight'
#             elif 'DQN' in traffic_performance[traffic_name][ti][1][1:]:
#                 name='DQN'
#             else:
#                 name='SAC-GNN'
#             if name=='SAC-GNN Constrained':
#                 ax[0].plot(traffic_performance[traffic_name][ti][0], label=name, linewidth=2)
#             else:
#                 ax[0].plot(traffic_performance[traffic_name][ti][0],'--', label=name, linewidth=2)                
#             validation_duration = traffic_performance[traffic_name][ti][0][-validation_duration_length:]
#             final_duration = np.round(np.mean(validation_duration), decimals=2)
#             if len(np.where(traffic_performance[traffic_name][ti][0] == NAN_LABEL)[0]) == 0:
#                 # and len(traffic_performance[traffic_name][ti][0]) == num_rounds:
#                 tmp = traffic_performance[traffic_name][ti][0]
#                 if len(tmp) < num_rounds:
#                     tmp.extend([float("nan")] * (num_rounds - len(traffic_performance[traffic_name][ti][0])))
#                 performance_tmp.append(tmp)
#                 check_list.append(final_duration)
#             else:
#                 print("the length of traffic {} is shorter than {}".format(traffic_name, num_rounds))
#         ax[0].legend(bbox_to_anchor=(0.5, 1.25), loc='upper center', ncol=2, fancybox=True, fontsize=14)
#         ax[0].set_ylabel(ylabel=y_label, fontsize=14)
#         ax[0].set_xlabel(xlabel='Episode', fontsize=14)
#         check_list = np.array(check_list)
#         for ci in np.where(check_list > anomaly_threshold * np.mean(check_list))[0][::-1]:
#             del performance_tmp[ci]
#             print("anomaly traffic_name:{} id:{} err:{}".format(traffic_name, ci, check_list[ci] - np.mean(check_list)))
#         if len(performance_tmp) == 0:
#             print("The result of {} is not enough for analysis.".format(traffic_name))
#             continue
#         try:
#             performance_summary = np.array(performance_tmp)
#             print(traffic_name, performance_summary.shape)
#             ax[1].errorbar(x=range(len(traffic_performance[traffic_name][0][0])),
#                            y=np.mean(performance_summary, axis=0),
#                            yerr=np.std(performance_summary, axis=0))

#             psm = np.mean(performance_summary, axis=0)
#             validation_duration = psm[-validation_duration_length:]
#             final_duration = np.round(np.mean(validation_duration), decimals=2)

#             convergence = {1.2: len(psm) - 1, 1.1: len(psm) - 1}
#             for j in range(minimum_round, len(psm)):
#                 for level in [1.2, 1.1]:
#                     if max(psm[j:]) <= level * final_duration:
#                         if convergence[level] > j:
#                             convergence[level] = j
#             ax[1].plot([0, len(psm)], [final_duration, final_duration], linewidth=2, color="g")
#             ax[1].text(len(psm), final_duration * 2, "final-" + str(final_duration))
#             ax[1].plot([convergence[1.2], convergence[1.2]], [psm[convergence[1.2]], psm[convergence[1.2]] * 3],
#                        linewidth=2, color="b")
#             ax[1].text(convergence[1.2], psm[convergence[1.2]] * 2, "conv 1.2-" + str(convergence[1.2]))
#             ax[1].plot([convergence[1.1], convergence[1.1]], [psm[convergence[1.1]], psm[convergence[1.1]] * 3],
#                        linewidth=2, color="b")
#             ax[1].text(convergence[1.1], psm[convergence[1.1]] * 2, "conv 1.1-" + str(convergence[1.1]))
#             ax[1].set_title(traffic_name)
#             plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.3)
#             plt.savefig(figure_dir + "/" + traffic_name + "-" + mode_name + ".png")
#             plt.close()
#         except:
#             print("plot error")


def plot_segment_duration(round_summary, path, mode_name):
    save_path = os.path.join(path, "segments")
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    for key in round_summary.keys():
        if "duration" in key:
            f, ax = plt.subplots(1, 1, figsize=(12, 9), dpi=100)
            ax.plot(round_summary[key], linewidth=2, color='k')
            ax.set_title(key)
            plt.savefig(save_path + "/" + key + "-" + mode_name + ".png")
            plt.close()


def padding_duration(performance_duration):
    for traffic_name in performance_duration.keys():
        max_duration_length = max([len(x[0]) for x in performance_duration[traffic_name]])
        for i, ti in enumerate(performance_duration[traffic_name]):
            performance_duration[traffic_name][i][0].extend((max_duration_length - len(ti[0]))*[ti[0][-1]])

    return performance_duration


def performance_at_min_duration_round_plot(performance_at_min_duration_round, figure_dir, mode_name):
    for traffic_name in performance_at_min_duration_round:
        f, ax = plt.subplots(1, 1, figsize=(12, 9), dpi=100)
        for ti in range(len(performance_at_min_duration_round[traffic_name])):
            ax.plot(performance_at_min_duration_round[traffic_name][ti][0], linewidth=2)
        plt.savefig(figure_dir + "/" + "min_duration_round" + "-" + mode_name + ".png")
        plt.close()


def summary_detail_test(memo, total_summary):
    # each_round_train_duration
    
    performance_queue={}
    performance_duration = {}
    performance_emission = {}
    performance_speed={}
    performance_emission_params={}
    performance_peak_green_time = {}
    performance_peak_emission= {}
    performance_througput_phase={}
    performance_througput={}
    performance_at_min_duration_round = {}

    records_dir = os.path.join("records", memo)

    for traffic_file in os.listdir(records_dir):
        if ".xml" not in traffic_file and ".json" not in traffic_file:
            continue
        file='anon_4_4_hangzhou_real_5816.json_'
        # file='anon_3_4_jinan_real.json_'
        # file='anon_3_4_jinan_real_new.json_'
        # file='anon_3_3_300_0.3_bi_high.json_'
        # file='anon_1_1_2400_uni.json_'
        # if traffic_file not in [file+'SAC_Attention_Const200', file+'SAC_Attention200', file+'Colight_torch200', file+'DQN_torch200_old']:
        models=[file+'SAC_Attention_Const200_new_0201', file+'SAC_Attention200_new_0201', file+'Colight_torch200_new_0201', file+'DQN_torch200_new_0201']#,
                # file+'SAC_Const200_new', file+'SAC200_new']
        if traffic_file not in models:
        # if traffic_file not in [file+'02_01_10_49_48']:
        # if traffic_file not in [file+'SAC_Const', file+'SAC', file+'DQN_torch']:
        # if traffic_file not in [file+'MaxPressure', file+'Fixedtime']:#, 'anon_3_3_300_0.3_uni.json_SAC_Attention_Const']:
            continue

        print(traffic_file)

        min_queue_length = min_duration = min_duration2 = float('inf')
        min_queue_length_id = min_duration_ind = 0

        # get run_counts to calculate the queue_length each second
        exp_conf = open(os.path.join(records_dir, traffic_file, "exp.conf"), 'r')
        dic_exp_conf = json.load(exp_conf)
        print(dic_exp_conf)


        traffic_env_conf = open(os.path.join(records_dir, traffic_file, "traffic_env.conf"), 'r')
        dic_traffic_env_conf = json.load(traffic_env_conf)
        run_counts = dic_exp_conf["RUN_COUNTS"]
        num_rounds = dic_exp_conf["NUM_ROUNDS"]
        time_interval = 120
        num_seg = run_counts//time_interval
        num_intersection = dic_traffic_env_conf['NUM_INTERSECTIONS']


        traffic_vol = get_traffic_volume(dic_exp_conf["TRAFFIC_FILE"][0], run_counts)
        nan_thres = 120

        duration_each_round_list = []
        queue_length_each_round_list = []
        emission_each_round_list = []
        speed_each_round_list=[]
        peak_green_violation_each_round_list=[]
        peak_emission_violation_each_round_list = []
        num_vehicle_left_each_round_list = []
        throughput_by_phase_round_list = []
        throughput_all_round_list = []
        emission_params_round_list = []
        num_of_vehicle_in = []
        num_of_vehicle_out = []
        reg_params_round=[]
        
        train_round_dir = os.path.join(records_dir, traffic_file, "test_round")
        try:
            round_files = os.listdir(train_round_dir)
        except:
            print("no test round in {}".format(traffic_file))
            continue
        round_files = [f for f in round_files if "round" in f]
        round_files.sort(key=lambda x: int(x[6:]))

        round_summary = {}
        for round in round_files:
            print("===={0}".format(round))

            df_vehicle_all = []
            queue_length_each_round = []
            emission_each_round = []
            speed_each_round=[]
            peak_max_green_violations_round=[]
            peak_emission_violations_round = []
            throughput_all_round = []
            num_vehicle_left_round = []
            throughput_by_phase_round = {i:0 for i in range(4)}

            list_duration_seg = [float('inf')] * num_seg
            list_queue_length_seg = [float('inf')] * num_seg
            list_queue_length_id_seg = [0] * num_seg
            list_duration_id_seg = [0] * num_seg
            list_num_vehicle_inter=[]
            list_emission_inter=[]
            for inter_index in range(num_intersection):

                try:

                    round_dir = os.path.join(train_round_dir, round)
                    


                    # summary items (queue_length) from pickle
                    f = open(os.path.join(round_dir, "inter_{0}.pkl".format(inter_index)), "rb")
                    samples = pkl.load(f)
                    queue_length_each_inter_each_round = 0
                    list_num_vehicle_sample=[]
                    list_emission_sample=[]
                    list_vehicle_speed=[]
                    for sample in samples:
                        queue_length_each_inter_each_round += sum(sample['state']['lane_num_vehicle_been_stopped_thres1'])
                        list_num_vehicle_sample.append(sum(sample['state']['lane_num_vehicle']))
                        list_emission_sample.append(sum(sample['state']['lane_vehicle_emission']))
                        list_vehicle_speed.append(np.mean(sample['state']['vehicle_speed_img']))
                    queue_length_each_inter_each_round = queue_length_each_inter_each_round//len(samples)
                    peak_max_green_violations=samples[-1]['state']['peak_max_green_violations']
                    peak_emission_violations=samples[-1]['state']['peak_emission_violations']
                    num_vehicle_left = samples[-1]['state']['num_vehicle_left']
                    throughput_by_phase = samples[-1]['state']['throughput_by_phase']
                    throughput_all = sum([samples[-1]['state']['throughput_by_phase'][i] for i in range(4)])
                    
                    
                    f.close()

                    # summary items (duration) from csv
                    df_vehicle_inter = pd.read_csv(os.path.join(round_dir, "vehicle_inter_{0}.csv".format(inter_index)),
                                                     sep=',', header=0, dtype={0: str, 1: float, 2: float},
                                                     names=["vehicle_id", "enter_time", "leave_time"])
                    df_vehicle_inter['leave_time_origin'] = df_vehicle_inter['leave_time']
                    df_vehicle_inter['leave_time'].fillna(run_counts,inplace=True)
                    df_vehicle_inter['duration'] = df_vehicle_inter["leave_time"].values - df_vehicle_inter["enter_time"].values
                    ave_duration = df_vehicle_inter['duration'].mean(skipna=True)
                    print("------------- inter_index: {0}\tave_duration: {1}\tave_queue_length: {2}"
                          .format(inter_index, ave_duration, queue_length_each_inter_each_round))

                    df_vehicle_all.append(df_vehicle_inter)
                    queue_length_each_round.append(queue_length_each_inter_each_round)
                    emission_each_round.append(sum(list_emission_sample))
                    speed_each_round.append(np.mean(list_vehicle_speed))
                    peak_max_green_violations_round.append(peak_max_green_violations)
                    peak_emission_violations_round.append(peak_emission_violations)
                    throughput_all_round.append(throughput_all)
                    num_vehicle_left_round.append(num_vehicle_left)
                    for i in range(4):
                        throughput_by_phase_round[i]+=throughput_by_phase[i]
                    list_num_vehicle_inter.append(list_num_vehicle_sample)
                    list_emission_inter.append(list_emission_sample)
                except:
                    queue_length_each_round.append(NAN_LABEL)
                    # num_of_vehicle_in.append(NAN_LABEL)
                    # num_of_vehicle_out.append(NAN_LABEL)

            if len(df_vehicle_all)==0:
                print("====================================EMPTY")
                continue

            df_vehicle_all = pd.concat(df_vehicle_all)
            vehicle_duration = df_vehicle_all.groupby(by=['vehicle_id'])['duration'].sum()
            ave_duration = vehicle_duration.mean()
            ave_queue_length = np.mean(queue_length_each_round)
            ave_emission = np.mean(emission_each_round)
            ave_speed = np.mean(speed_each_round)
            peak_max_green = np.sum(peak_max_green_violations_round)
            peak_emission = np.sum(peak_emission_violations_round)
            throughput = np.sum(throughput_all_round)

            duration_each_round_list.append(ave_duration)
            queue_length_each_round_list.append(ave_queue_length)
            emission_each_round_list.append(ave_emission)
            speed_each_round_list.append(ave_speed)
            peak_green_violation_each_round_list.append(peak_max_green)
            peak_emission_violation_each_round_list.append(peak_emission)
            num_vehicle_left_each_round_list.append(num_vehicle_left_round)
            throughput_by_phase_round_list.append(throughput_by_phase_round)
            throughput_all_round_list.append(throughput)
            reg_params=[]
            for i in range(num_intersection):
                num_vehicle=np.array(list_num_vehicle_inter[i]).reshape(-1,1)
                emission=np.array(list_emission_inter[i]).reshape(-1,1)
                reg = LinearRegression().fit(num_vehicle, emission)
                reg_params.append([reg.coef_[0][0], reg.intercept_[0]])
            reg_params_round.append(reg_params)
            
            
            num_of_vehicle_in.append(len(df_vehicle_all['vehicle_id'].unique()))
            num_of_vehicle_out.append(len(df_vehicle_all.dropna()['vehicle_id'].unique()))

            print("==== round: {0}\tave_duration: {1}\tave_queue_length_per_intersection:{2}\tave_emission:{5}"
                  "num_of_vehicle_in:{3}\tnum_of_vehicle_out:{4}"
                  .format(round, ave_duration,ave_queue_length,num_of_vehicle_in[-1],num_of_vehicle_out[-1], ave_emission))

            duration_flow = vehicle_duration.reset_index()

            duration_flow['direction'] = duration_flow['vehicle_id'].apply(lambda x:x.split('_')[1])
            duration_flow_ave = duration_flow.groupby(by=['direction'])['duration'].mean()
            print(duration_flow_ave)

            # print(real_traffic_vol, traffic_vol, traffic_vol - real_traffic_vol, nan_num)
            if min_queue_length > ave_queue_length:
                min_queue_length = np.mean(queue_length_each_round)
                min_queue_length_id = int(round[6:])
            #
            # valid_flag = json.load(open(os.path.join(round_dir, "valid_flag.json")))
            # if valid_flag['0']:  # temporary for one intersection
            #     if min_duration > ave_duration and ave_duration > 24:
            #         min_duration = ave_duration
            #         min_duration_ind = int(round[6:])


            #### This is for long time

            if num_seg > 1:
                for i, interval in enumerate(range(0, run_counts, time_interval)):
                    did = df_vehicle_all[(df_vehicle_all["enter_time"]< interval+time_interval) &
                                         (df_vehicle_all["enter_time"].values > interval)]
                    #vehicle_in_seg = sum([int(x) for x in (df_vehicle_inter_0["enter_time"][did].values > 0)])
                    #vehicle_out_seg = sum([int(x) for x in (df_vehicle_inter_0["leave_time"][did].values > 0)])

                    vehicle_duration_seg = did.groupby(by=['vehicle_id'])['duration'].sum()
                    ave_duration_seg = vehicle_duration_seg[vehicle_duration_seg>10].mean()
                    # print(traffic_file, round, i, ave_duration)
                    # real_traffic_vol_seg = 0
                    # nan_num_seg = 0
                    # for time in duration_seg:
                    #     if not isnan(time):
                    #         real_traffic_vol_seg += 1
                    #     else:
                    #         nan_num_seg += 1

                    # print(real_traffic_vol, traffic_vol, traffic_vol - real_traffic_vol, nan_num)
                    nan_num_seg = did['leave_time_origin'].isna().sum()

                    if nan_num_seg < nan_thres:
                        list_duration_seg[i] = ave_duration_seg
                        list_duration_id_seg[i] = int(round[6:])

                    #round_summary = {}
                for j in range(num_seg):
                    key = "min_duration-" + str(j)

                    if key not in round_summary.keys():
                        round_summary[key] = [list_duration_seg[j]]
                    else:
                        round_summary[key].append(list_duration_seg[j])
                #round_result_dir = os.path.join("summary", memo, traffic_file)
                #if not os.path.exists(round_result_dir):
                #    os.makedirs(round_result_dir)

        
        # result_dir = os.path.join(records_dir, traffic_file)
        result_dir = os.path.join("summary", memo, traffic_file)
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
        _res = {
            "duration": duration_each_round_list,
            "queue_length": queue_length_each_round_list,
            "emission": emission_each_round_list,
            "peak_green_violation":peak_green_violation_each_round_list,
            "peak_emission_violation":peak_emission_violation_each_round_list,
            "vehicle_in": num_of_vehicle_in,
            "vehicle_out": num_of_vehicle_out
        }
        result = pd.DataFrame(_res)
        result.to_csv(os.path.join(result_dir, "test_results.csv"))

        if ".xml" in traffic_file:
            traffic_name, traffic_time = traffic_file.split(".xml")
        elif ".json" in traffic_file:
            traffic_name, traffic_time = traffic_file.split(".json")

        if num_seg > 1:
            round_result = pd.DataFrame(round_summary)
            round_result.to_csv(os.path.join(result_dir, "test_seg_results.csv"), index=False)
            plot_segment_duration(round_summary, result_dir, mode_name="test")
            duration_each_segment_list = round_result.iloc[min_duration_ind][1:].values

            if traffic_name not in performance_at_min_duration_round:
                performance_at_min_duration_round[traffic_name] = [(duration_each_segment_list, traffic_time)]
            else:
                performance_at_min_duration_round[traffic_name].append((duration_each_segment_list, traffic_time))


        # print(os.path.join(result_dir, "test_results.csv"))

        # total_summary
        total_summary = get_metrics(duration_each_round_list, queue_length_each_round_list,
                                    min_duration, min_duration_ind, min_queue_length, min_queue_length_id,
                                    traffic_file, total_summary,
                                    mode_name="test", save_path=result_dir, num_rounds=num_rounds,
                                    min_duration2=None if "peak" not in traffic_file else min_duration2)
        # peak_green_violation_each_round_list =[i if i<10000 else 10000 for i in peak_green_violation_each_round_list]
        # peak_green_violation_each_round_list =[i if i<4000 else 4000 for i in peak_green_violation_each_round_list]
        if traffic_name not in performance_duration:
            performance_duration[traffic_name] = [(duration_each_round_list, traffic_time)]
            performance_queue[traffic_name] = [(queue_length_each_round_list, traffic_time)]
            performance_emission[traffic_name] = [(emission_each_round_list, traffic_time)]
            performance_speed[traffic_name] = [(speed_each_round_list, traffic_time)]
            performance_peak_green_time[traffic_name] = [(peak_green_violation_each_round_list, traffic_time)]
            performance_peak_emission[traffic_name] = [(peak_emission_violation_each_round_list, traffic_time)]
            performance_througput_phase[traffic_name] = [(throughput_by_phase_round_list, traffic_time)]
            performance_througput[traffic_name] = [(throughput_all_round_list, traffic_time)]
        else:
            performance_duration[traffic_name].append((duration_each_round_list, traffic_time))
            performance_queue[traffic_name].append((queue_length_each_round_list, traffic_time))
            performance_emission[traffic_name].append((emission_each_round_list, traffic_time))
            performance_speed[traffic_name].append((speed_each_round_list, traffic_time))
            performance_peak_green_time[traffic_name].append((peak_green_violation_each_round_list, traffic_time))
            performance_peak_emission[traffic_name].append((peak_emission_violation_each_round_list, traffic_time))
            performance_througput_phase[traffic_name].append((throughput_by_phase_round_list, traffic_time))
            performance_througput[traffic_name].append((throughput_all_round_list, traffic_time))
            



        total_result = pd.DataFrame(total_summary)
        total_result.to_csv(os.path.join("summary", memo, "total_test_results.csv"))

    # duration_max_pressure=[126.8239]*len(duration_each_round_list)
    # performance_duration[traffic_name].append((duration_max_pressure, '_max_pressure'))
    
    # duration_fixedtime=[326.6513]*len(duration_each_round_list)
    # performance_duration[traffic_name].append((duration_fixedtime, '_fixedtime'))
    
    
    
    figure_dir = os.path.join("summary", memo, "figures")
    if not os.path.exists(figure_dir):
        os.makedirs(figure_dir)
    if dic_exp_conf["EARLY_STOP"]:
        performance_duration = padding_duration(performance_duration)
    summary_plot(performance_duration, figure_dir, mode_name="test", num_rounds=num_rounds, y_label='Travel Time (s)')
    summary_plot(performance_emission, figure_dir, mode_name="test_emission", num_rounds=num_rounds, y_label='$CO_2$ Emission rate (liter)')
    summary_plot(performance_peak_green_time, figure_dir, mode_name="Peak_green_violation", num_rounds=num_rounds, y_label='Green Violation')
    summary_plot(performance_peak_emission, figure_dir, mode_name="Peak_emission_violation", num_rounds=num_rounds, y_label='Emission Violation')
    summary_plot(performance_througput, figure_dir, mode_name="Througput", num_rounds=num_rounds, y_label='Throughput')

    summary_plot_throughput_phase(performance_througput_phase, figure_dir, mode_name="Througput_phase", num_rounds=num_rounds, y_label='Throughput')
    # performance_at_min_duration_round_plot(performance_at_min_duration_round, figure_dir, mode_name="test")
    
    
def regression_params(emission_params_methods):
    
    for traffic_name in emission_params_methods:
        emission_params=emission_params_methods[traffic_name]
        regression_params_round=[]
        for r in range(len(emission_params)):
            num_int=emission_params[r]['num_vehicle'].shape[1]
            reg_params=[]
            for i in range(num_int):
                num_vehicle=emission_params[r]['num_vehicle'][:,i]
                emission=emission_params[r]['emission'][:,i]
                reg = LinearRegression().fit(num_vehicle, emission)
                reg_params[i].append([reg.coef_[0], reg.intercept_[0]])
            regression_params_round.append(regression_params)
        
    return


##TODO multi-intersection
def summary_detail_baseline(memo):

    DETAIL_ARTERIAL = True
    total_summary = []

    records_dir = os.path.join("records", memo)
    for traffic_file in os.listdir(records_dir):
        ANON_ENV = False


        if ".xml" not in traffic_file and "anon" not in traffic_file:
            continue
        if "anon" in traffic_file:
            ANON_ENV = True

        exp_conf = open(os.path.join(records_dir, traffic_file, "exp.conf"), 'r')
        dic_exp_conf = json.load(exp_conf)
        run_counts = dic_exp_conf["RUN_COUNTS"]

        avg_pressure = 0

        print(traffic_file)

        train_dir = os.path.join(records_dir, traffic_file)

        if os.path.getsize(os.path.join(train_dir, "inter_0.pkl")) > 0:
            with open(os.path.join(records_dir, traffic_file, 'agent.conf'), 'r') as agent_conf:
                dic_agent_conf = json.load(agent_conf)

            df_vehicle = []
            NUM_OF_INTERSECTIONS = int(traffic_file.split('_')[1])*int(traffic_file.split('_')[2])

            list_f = ["inter_%d.pkl" % i for i in range(int(NUM_OF_INTERSECTIONS))]

            for f in list_f:
                pressure_each_inter = 0

                node_index = f.split('inter_')[1].split('.pkl')[0]
                print("node",node_index)
                f = open(os.path.join(train_dir, f), "rb")
                samples = pkl.load(f)
                for sample in samples:
                    pressure_each_inter += sum((sample['state']['lane_num_vehicle_been_stopped_thres1']))
                f.close()

                pressure_each_inter = pressure_each_inter/len(samples)
                avg_pressure += pressure_each_inter

                vehicle_csv = "vehicle_inter_{0}.csv".format(node_index)

                df_vehicle_inter_0 = pd.read_csv(os.path.join(train_dir, vehicle_csv),
                                                 sep=',', header=0, dtype={0: str, 1: float, 2: float},
                                                 names=["vehicle_id", "enter_time", "leave_time"])

                # print(df_vehicle_inter_0)

                if ANON_ENV:
                    flow_car = pd.DataFrame(df_vehicle_inter_0['vehicle_id'].str.split('_', -1).tolist(), columns=['flow','flow_id', 'car_id'])
                else:
                    flow_car = pd.DataFrame(df_vehicle_inter_0['vehicle_id'].str.split('.', 1).tolist(), columns=['flow_id', 'car_id'])
                df_vehicle_inter_0 = pd.concat([flow_car, df_vehicle_inter_0], axis=1)
                df_vehicle_inter_0.fillna(run_counts,inplace=True)
                df_vehicle_inter_0['duration'] = df_vehicle_inter_0["leave_time"] - df_vehicle_inter_0["enter_time"]


                df_vehicle.append(df_vehicle_inter_0)
                print(df_vehicle_inter_0.groupby(['flow_id'])['duration'].mean()) # mean for every intersection

            df_vehicle = pd.concat(df_vehicle,axis=0)

            flow_df = df_vehicle.groupby(['flow_id', 'car_id']).sum()
            arterial_duration = 0
            side_street_duration = 0
            if DETAIL_ARTERIAL:
                detail_arterial = flow_df.groupby('flow_id').mean()
                save_path = os.path.join('records',memo, traffic_file).replace("records","summary")
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                detail_arterial.to_csv(os.path.join(save_path, 'flow.csv'))
                arterial_duration = np.average(detail_arterial[:2])
                side_street_duration = np.average(detail_arterial[3:])
                avg_pressure = avg_pressure/NUM_OF_INTERSECTIONS


            car_num_out_df = df_vehicle.groupby(by=['flow_id', 'car_id'])['leave_time'].apply(lambda x: x.shape[0] != x.count())
            car_num_out = car_num_out_df[car_num_out_df].count()

            ave_duration_all = flow_df['duration'].mean()
            total_summary.append([traffic_file,ave_duration_all, avg_pressure,flow_df.shape[0],car_num_out,dic_agent_conf["FIXED_TIME"],arterial_duration,side_street_duration])
        else:
            shutil.rmtree(train_dir)

    total_summary = pd.DataFrame(total_summary)
    total_summary.sort_values([0], ascending=[True], inplace=True)
    total_summary.columns = ['TRAFFIC','DURATION','PRESSURE','CAR_NUMBER_IN','CAR_NUMBER_OUT','CONFIG','ARTERIAL','SIDE_STREET']
    total_summary.to_csv(os.path.join("records", memo, "total_baseline_results.txt").replace("records", "summary"),sep='\t',index=False)



# def main(memo=None):
#     total_summary = {
#         "traffic": [],
#         "traffic_file": [],
#         "min_queue_length": [],
#         "min_queue_length_round": [],
#         "min_duration": [],
#         "min_duration_round": [],
#         "final_duration": [],
#         "final_duration_std": [],
#         "convergence_1.2": [],
#         "convergence_1.1": [],
#         "nan_count": [],
#         "min_duration2": []
#     }
#     if not memo:
#         memo = "pipeline_500"
#     #summary_detail_train(memo, copy.deepcopy(total_summary))
#     summary_detail_test(memo, copy.deepcopy(total_summary))
#     # summary_detail_test_segments(memo, copy.deepcopy(total_summary))

def print_samples(memo):
    records_dir = os.path.join("records", memo)

    for traffic_file in os.listdir(records_dir):
        if ".xml" not in traffic_file:
            continue
        print(traffic_file)
        exp_conf = open(os.path.join(records_dir, traffic_file, "exp.conf"), 'r')
        dic_exp_conf = json.load(exp_conf)
        run_counts = dic_exp_conf["RUN_COUNTS"]
        num_rounds = dic_exp_conf["NUM_ROUNDS"]
        num_seg = run_counts // 3600
        train_round_dir = os.path.join(records_dir, traffic_file, "train_round")
        round_files = os.listdir(train_round_dir)
        round_files = [f for f in round_files if "round" in f]
        round_files.sort(key=lambda x: int(x[6:]))
        for round in round_files:
            print(round)
            try:
                round_dir = os.path.join(train_round_dir, round)
                for gen in os.listdir(round_dir):
                    if "generator" not in gen:
                        continue

                    gen_dir = os.path.join(records_dir, traffic_file, "train_round", round, gen)
                    f = open(os.path.join(gen_dir, "inter_0.pkl"), "rb")
                    samples = pkl.load(f)

                    for sample in samples:
                        print("action", sample['action'], end='')
                        print("time", sample['time'], end='')
                        for key in sample['state'].keys():
                            if sample['state'][key] != None:
                                print(key, sample['state'][key], end='')

                        print('\n')
                    f.close()
            except:
                print("NO SAMPLES! round", round)


def print_samples():
    # gen_dir="/Users/chenchacha/RLSignal/records/test_pressure/2_intersections_uniform_300_0.3_uni.xml_12_20_03_14_11/train_round/round_7/generator_0"
    #
    #
    # f = open(os.path.join(gen_dir, "inter_0.pkl"), "rb")
    # samples = pkl.load(f)
    #
    # for sample in samples:
    #     print("action", sample['action'], end='\t')
    #     print("time", sample['time'], end='\t')
    #     for key in sample['state'].keys():
    #         if sample['state'][key] != None:
    #             print(key, sample['state'][key], end='')
    #
    #     print('\n')
    # f.close()

    f=open(os.getcwd()+"/records/0515_afternoon_Colight_6_6_bi/anon_6_6_300_0.3_bi.json_11_03_13_20_47/train_round/total_samples_inter_35.pkl","rb")
    samples = []
    try:
        while 1:
            samples.extend(pkl.load(f))
    except:
        pass
    # samples = (pkl.load(f))
    print(len(samples))
    count = 0
    for round in samples:
        print(count)
        count +=1
        for sample in round:
            print(sample)
    #for sample in samples:
    #    print(sample)
    f.close()


if __name__ == "__main__":
    total_summary = {
        "traffic": [],
        "inter_num":[],
        "traffic_volume":[],
        "ratio":[],
        "min_queue_length": [],
        "min_queue_length_round": [],
        "min_duration": [],
        "min_duration_round": [],
        "final_duration": [],
        "final_duration_std": [],
        "convergence_1.2": [],
        "convergence_1.1": [],
        "nan_count": [],
        "min_duration2": []
    }




    memo = "multi_phase/multi_phase_12_12_600_700_layer_10"

    args = parse_args()

    # print_samples()
    if args.b:
        summary_detail_baseline(args.memo)
    else:
        summary_detail_test(args.memo, copy.deepcopy(total_summary))
        
#fig, ax = plt.subplots(figsize=(10,8))
#ax.scatter(num_vehicle, emission, marker='o', linestyle='--', label='raw', color='b', linewidth=2)
# # ax.plot(num_veh_sort, total_emission_pred, marker='o', linestyle='--', label='pred', color='y', linewidth=2)
# # ax.plot(epsi, twoweeks, marker='o', linestyle='--', label='6-days', color='k', linewidth=2)
#plt.grid(axis='y', alpha=0.75)
# # plt.xticks(epsi, eps_all, fontsize=14)
# #plt.yticks( fontsize=14) 
# # legend = ax.legend(fontsize='x-large')
# #plt.xticks([0, 1,2,3, 4], methods)
#plt.ylabel('Total Emission', fontsize=16)
#plt.xlabel('Number of vehicle', fontsize=16)
# #plt.title('Moved links over '+str(len(vmt_clean)*2)+' links for '+city)
# # plt.savefig('Figures_1129/link_count_change.png', dpi=500)
#plt.show()




