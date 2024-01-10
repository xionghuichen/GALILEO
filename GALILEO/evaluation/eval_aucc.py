import os
import sys

sys.path.append(os.path.join(os.getcwd()))

from collections import defaultdict
import pickle
import h5py
import numpy as np
import matplotlib.pyplot as plt
import random
import pandas as pd
import torch
import argparse

from supervised_learning import Mlp

gpu_index = 0
n_continuous_y = 1
n_onehot_y = 2
n_y = 3
n_exclude = 8
n_treatment = 2
n_onehot_state = 10
n_continuous_state = 93
n_state = 113


def std_mean(state, mean=None, std=None):
    if mean is None or std is None:
        mean = state.mean(0)
        std = state.std(0, unbiased=False)
    state = (state - mean) / (std + 1e-7)

    return state, mean, std


def min_max(state, min=None, max=None):
    if min is None or max is None:
        min = state.min(0).values
        max = state.max(0).values

    state = (state - min) / (max - min)
    state = torch.clamp(state, 0, 1)

    return state, min, max


def normalization(state, dim_con, down, up, normal_type='min_max'):
    """
    """
    one_hot = state[:, dim_con:]
    continuous = state[:, :dim_con]
    if normal_type == 'min_max':
        continuous, down, up = min_max(continuous, down, up)
    elif normal_type == 'mean_std':
        continuous, down, up = std_mean(continuous, down, up)
    else:
        raise ValueError('Wrong normal type')
    state = torch.cat([continuous, one_hot], dim=-1)

    return state, down, up


def load_data(data_path):
    data = h5py.File(data_path)
    return data


def load_model(model_path, data, model_type, predict_type):
    device = f'cuda:{gpu_index}' if torch.cuda.is_available() else 'cpu'

    if "sl" in model_path or "ipw" in model_path:
        state_action_shape = data[f"real_{predict_type}_data"]["static_obs"][:].shape[1] + \
                             data[f"real_{predict_type}_data"]["action"][:].shape[1]
        y_shape = data[f"real_{predict_type}_data"]["y"][:].shape[1]

        # load baseline model
        model = Mlp(device, n_exclude, n_continuous_y, state_action_shape, y_shape, 256).to(torch.device(device))
        cpt = torch.load(model_path)
        model.load_state_dict(cpt)
        model.eval()
    else:
        # load revive model
        model = pickle.load(open(model_path, "rb"), encoding="utf-8")
        model.to(device)
    return model


# split batch
def get_batch_indices(total_indices, batch_size, shuffle=False):
    if shuffle:
        random.seed(0)
        random.shuffle(total_indices)

    start, end = 0, batch_size
    while True:

        yield total_indices[start:min(end, len(total_indices))]

        if end >= len(total_indices):
            break

        start += batch_size
        end += batch_size

    # 根据model对数据预处理


def preprocess_data(data, model_type, predict_type="grab", exp_type="+"):
    # random money为正和为负都要画图
    if "random_money" in data.keys():
        exp_money = data["random_money"][:][:, 1]  # 这里应该index=1是实验加的钱
        if exp_type == "+":
            total_indices = np.where(exp_money >= 0)[0]
        else:
            total_indices = np.where(exp_money < 0)[0]
    else:
        total_indices = np.arange(len(data["action"][:]))

    # 如果预测arrived time或者达标率，需要只用接单的数据
    # 这里baseline应该对所有数据都forward，但是后处理的时候分开
    if predict_type == "arrived_time" or predict_type == "get_youjiangsong_reward":
        y = data["y"][:][:, 1]
        terminal_indices = np.where(y == 1)[0]  # TODO: 这里还要排出达标率-1的样本

        total_indices = np.intersect1d(terminal_indices, total_indices)

        y = data["y"][:, -1]
        terminal_indices = np.where(y != -1)[0]
        total_indices = np.intersect1d(terminal_indices, total_indices)

    # 把原始数据过滤
    static_obs = data["static_obs"][:][total_indices]
    action = data["action"][:][total_indices]
    y = data["y"][:][total_indices]

    output = {}
    if "random_money" in data.keys():
        random_money = data["random_money"][:][:, 1].reshape(-1)
        random_money = random_money[total_indices]
        output["random_money"] = random_money
        assert len(random_money) == len(action)

    output.update({
        "action": action,
        "y": y,
        "static_obs": static_obs
    })
    assert len(action) == len(y) == len(static_obs)

    if model_type == "revive":
        if predict_type == "grab":
            dynamic_obs = data["dynamic_obs"][:][total_indices]
            assert len(dynamic_obs) == len(action)
            output.update({"dynamic_obs": dynamic_obs})
            return output

        return output

    # baseline model
    static_obs, action, y = torch.from_numpy(static_obs), torch.from_numpy(action), torch.from_numpy(y)
    static_obs, s_mean, s_std = normalization(static_obs, n_continuous_state, None, None, "mean_std")
    action, action_mean, action_std = normalization(action, 2, None, None, "mean_std")
    # y, y_min, y_max = normalization(y, 1, None, None, "min_max")  # todo: 这个需要归一化吗
    state_action = torch.cat([static_obs, action], dim=1)

    assert len(action) == len(state_action)
    output.update({"state_action": state_action})
    return output


# 将数据整理为模型的输入形式
def get_model_input(data, indices, model_type, predict_type, device=torch.device("cpu")):
    if model_type == "revive":
        if predict_type == "grab":
            action, dynamic_obs, static_obs = data["action"], data["dynamic_obs"], data["static_obs"]
            batch = {
                "dynamic_obs": dynamic_obs[indices],
                "static_obs": static_obs[indices],
                "action": action[indices]
            }
        else:
            action, static_obs = data["action"], data["static_obs"]
            batch = {
                "static_obs": static_obs[indices],
                "action": action[indices]
            }
    else:
        state_action = data["state_action"]
        batch = state_action[indices]
        batch = batch.to(device).float()
    return batch


# 将模型的输出整理为需要预测的格式
def postprocess_batch(output, model_type, predict_type):
    if model_type == "revive":
        if predict_type == "arrived_time":
            output = output["y2"][:, 0]  # TODO: 哪两维啊
        elif predict_type == "get_youjiangsong_reward":
            output = output["y2"][:, 1]  # 有奖送分类是二分类
        else:
            output = output["y1"][:, -1]  # 接单率是三分类
    else:
        if predict_type == "grab":
            output = output[1]
        elif predict_type == "arrived_time":
            output = output[0]
        else:
            output = output[2]

    if isinstance(output, torch.Tensor):
        output = output.cpu().numpy()
    if len(output.shape) == 1:
        output = output.reshape(-1, 1)
    return output


# 使用model对data数据进行预测，返回预测结果
def roll_out_model(model, raw_data, model_type, predict_type="grab", exp_type="+"):
    # model_type: revive, sl, ipw
    # predict_type: grab, arrived_time, 达标率
    # return: pred_label, real_money, real_label

    batch_size = 1024
    device = model.device
    if isinstance(device, str):
        device = torch.device(device)

    # prepare data
    data = preprocess_data(raw_data, model_type, predict_type, exp_type)
    total_indices = np.arange(len(data["action"]))

    # total_indices = np.random.choice(np.arange(len(data["action"])), 1024)  # debug
    # total_indices = np.arange(1024)

    # get forward function
    def forward_func(*args):
        if model_type == "revive":
            if predict_type == "arrived_time":
                return model.infer_one_step(*args)
            return model.infer_prob_one_step(*args)
        return model.forward(*args)

    # calculate prediction
    # 如果是revive model，只能单独预测；baseline model一起预测
    pred_dim = 1  # 如果predict_type是arrived time，会预测两维

    preds = np.zeros((len(total_indices), pred_dim))
    forward_indices = []
    for batch_indices in get_batch_indices(total_indices, batch_size):
        # model forward
        batch = get_model_input(data, batch_indices, model_type, predict_type, device)

        # model forward
        with torch.no_grad():
            pred = forward_func(batch)

        # post process data
        pred = postprocess_batch(pred, model_type, predict_type)

        preds[batch_indices] = pred
        forward_indices.extend(batch_indices.tolist())

    labels = data["y"][forward_indices]

    if predict_type == "grab":
        labels = labels[:, 1].reshape(-1, 1)
    elif predict_type == "arrived_time":
        labels = labels[:, 0].reshape(-1, 1)
    else:
        labels = labels[:, -1].reshape(-1, 1)

    if "random_money" in raw_data.keys():
        money = (data["random_money"][forward_indices]).reshape(-1, 1)
        assert len(money) == len(preds) == len(labels)
        return preds, money, labels
    else:
        assert len(preds) == len(labels)
        return preds, labels


# 计算样本的关于预测值pred的roi，并按照roi排序，返回排序后的money及
quantile_list = np.linspace(0, 1, 21)[1:]


def caculate_roi(base_pred, exp_pred, base_label, exp_label, money, sort=True, predict_type="grab", exp_type="+"):
    # base pred: 对照组使用model的预测值
    # exp_pred: 实验组使用model的预测值
    # label: 对照组和实验组真实的label（对应于pred）
    # money：实验组的实验money
    # return: money, delta labels (sorted)
    # roi = (exp_pred - base_pred) / (delta money)

    base_pred_mean = base_pred.mean(0)
    base_label_mean = base_label.mean(0)

    roi = (exp_pred - base_pred_mean) / (abs(money) + 1e-6)

    if predict_type == "arrived_time":
        roi = - roi

    if exp_type == "-":
        roi = - roi

    list_to_sort = list(zip(roi, money, exp_label))
    if sort:
        list_to_sort.sort(key=lambda x: x[0], reverse=True)  # x: roi, money, exp_label

    list_to_sort = np.array(list_to_sort)

    exp_start, base_start = 0, 0
    money = list_to_sort[:, 1]
    exp_label = list_to_sort[:, 2]  # TODO: base label需要排序吗

    money_list = np.zeros(len(quantile_list))
    y_list = np.zeros(len(quantile_list))
    for i in range(len(quantile_list)):
        exp_indices = np.arange(exp_start, int(quantile_list[i] * len(money)))
        base_indices = np.arange(base_start, int(quantile_list[i] * len(base_label)))

        tmp_exp_label = sum(exp_label[exp_indices])
        tmp_base_label = sum(base_label[base_indices]) / len(base_indices) * len(exp_indices)

        money_list[i] = sum(money[exp_indices])
        y_list[i] = tmp_exp_label - tmp_base_label
        exp_start = int(quantile_list[i] * len(money))
        base_start = int(quantile_list[i] * len(base_label))

    # if predict_type == "arrived_time":
    #     y_list = - y_list

    return money_list, y_list


def plot_aucc(money_list, label_list, algo_list, predict_type, exp_type, city=None, save_path=None):
    # makedirs
    if save_path is None:
        save_path = f"figs/{city}"
        if not os.path.exists(save_path):
            os.makedirs(save_path)
    save_path = os.path.join(save_path, f"{predict_type}_{exp_type}.png")

    # plot
    fig, ax = plt.subplots()

    area_list = []
    for i in range(len(algo_list)):
        x, y = np.abs(money_list[i]), label_list[i]
        x, y = np.cumsum(x), np.cumsum(y)
        algo = algo_list[i]
        if algo == "random":
            x = np.array([0, x[-1]])
            y = np.array([0, y[-1]])
        else:
            x = np.hstack([0, x])
            y = np.hstack([0, y])

        if algo == "revive":
            algo = "galileo"
        ax.plot(x, y, label=algo)

        # 计算面积
        area = 0
        pre_length = 0
        pre_x = 0
        for j in range(len(x)):
            cur_length = y[j]
            area += 0.5 * (pre_length + cur_length) * (x[j] - pre_x)
            pre_x = x[j]
            pre_length = y[j]
        area_list.append(area)

    ax.set_xlabel("money")
    ax.ticklabel_format(style='sci', scilimits=(-1, 2), axis='both')
    ax.set_ylabel(predict_type)
    ax.legend()

    fig.savefig(save_path)

    return area_list


def calculate_aucc(real_data, random_data, model, model_type, predict_type, save_path):
    results = defaultdict(float)
    for exp_type in ["+", "-"]:
        # calculate the base point of roi
        base_pred, base_label = roll_out_model(
            model, real_data, model_type=model_type, predict_type=predict_type, exp_type=exp_type)

        # calculate experimental data
        exp_pred, exp_money, exp_label = roll_out_model(
            model, random_data, model_type=model_type, predict_type=predict_type, exp_type=exp_type)

        # save roi results to results dict
        money, labels = caculate_roi(base_pred, exp_pred, base_label, exp_label, exp_money, sort=True,
                                     predict_type=predict_type, exp_type=exp_type)  # 横轴 纵轴
        random_money, random_labels = caculate_roi(base_pred, exp_pred, base_label, exp_label, exp_money, sort=False,
                                                   predict_type=predict_type, exp_type=exp_type)

        area_list = plot_aucc([money, random_money], [labels, random_labels], ["galio", "random"], predict_type,
                              exp_type, city="")
        a, b = area_list
        aucc = (a + b) / (2 * b)

        print(f"{model_type} - {predict_type} - {exp_type} : {aucc}")
        results[exp_type] = aucc

    return results


if __name__ == "__main__":
    aucc_dict = {}  # city predict_type exp_type model
    city_list = ["500100", "320100", "510100"]
    predict_type_list = ["get_youjiangsong_reward", "arrived_time", "grab"]
    exp_type_list = ["+", "-"]
    model_type_list = ["sl", "ipw", "revive"]
    for city in city_list:
        # sl_prefix = "" if city == "500100" else "sl_"
        sl_prefix = "sl_"
        base_path = "/mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-peisongpa/qixintong/" \
                    "youjiangsong/code/youjiangsong_v2/"
        path_dict = dict(
            # real data
            real_arrived_time_data_path=base_path + f"revive/data/real_data/real_data/{city}/" \
                                                    "1121_0104_real_y12_onestep.h5",
            real_grab_data_path=base_path + f"revive/data/real_data/real_data/{city}/" \
                                            "1121_0104_real_y.h5",
            # exprimental data
            exp_arrived_time_data_path=base_path + f"revive/data/random_data/random_data/{city}/" \
                                                   "1121_0104_random_y12_onestep.h5",
            exp_grab_data_path=base_path + f"revive/data/random_data/random_data/{city}/" \
                                           "1121_0104_random.h5",

            # model path
            arrived_time_revive_model_path=base_path + f"revive/logs/{city}_y12_onestep_batch2048_fintune20_gstep10/env.pkl",
            arrived_time_sl_model_path=base_path + f"sl_logs/{city}/{sl_prefix}{city}_seed10_revivedata/venv.pt",
            # ["{city}_seed10_revivedata", "{city}_seed1_revivedata", "{city}_revivedata"]
            arrived_time_ipw_model_path=base_path + f"ipw_logs/{city}/ipw_{city}_seed0_revivedata/venv.pt",
            # ipw_{city}_revivedata  ipw_{city}_seed10_revivedata  ipw_{city}_seed1_revivedata

            grab_revive_model_path=base_path + f"revive/logs/{city}_grab_batch2048_fintune20_gstep10_1/env.pkl",
            grab_sl_model_path=base_path + f"sl_logs/{city}/{sl_prefix}{city}_seed10_revivedata/venv.pt",
            # ["{city}_seed10_revivedata", "{city}_seed1_revivedata", "{city}_revivedata"]
            grab_ipw_model_path=base_path + f"ipw_logs/{city}/ipw_{city}_seed0_revivedata/venv.pt",
            # ipw_{city}_revivedata  ipw_{city}_seed10_revivedata  ipw_{city}_seed1_revivedata

            get_youjiangsong_reward_revive_model_path=base_path + f"revive/logs/{city}_y12_onestep_batch2048_fintune20_gstep10/env.pkl",
            get_youjiangsong_reward_sl_model_path=base_path + f"sl_logs/{city}/{sl_prefix}{city}_seed10_revivedata/venv.pt",
            # ["{city}_seed10_revivedata", "{city}_seed1_revivedata", "{city}_revivedata"]
            get_youjiangsong_reward_ipw_model_path=base_path + f"ipw_logs/{city}/ipw_{city}_seed0_revivedata/venv.pt",
            # ipw_{city}_revivedata  ipw_{city}_seed10_revivedata  ipw_{city}_seed1_revivedata
        )
        if city == "510100":
            path_dict[
                "grab_revive_model_path"] = base_path + f"revive/logs/510100_grab_batch2048_fintune20_gstep10/env.pkl"
        elif city == "320100":
            path_dict[
                "grab_revive_model_path"] = base_path + f"revive/logs/320100_grab_batch2048_fintune20_gstep10/env.pkl"
        elif city == "500100":
            path_dict[
                "arrived_time_revive_model_path"] = base_path + f"revive/logs/500100_y12_onestep_batch2048_fintune50_gstep10/env.pkl"

        # load data
        data = dict(
            real_arrived_time_data=load_data(path_dict["real_arrived_time_data_path"]),
            real_grab_data=load_data(path_dict["real_grab_data_path"]),
            exp_arrived_time_data=load_data(path_dict["exp_arrived_time_data_path"]),
            exp_grab_data=load_data(path_dict["exp_grab_data_path"]),
        )
        data["real_get_youjiangsong_reward_data"] = data["real_arrived_time_data"]
        data["exp_get_youjiangsong_reward_data"] = data["exp_arrived_time_data"]

        results = {
            "grab": {
                "sl": {"+": None, "-": None},
                "ipw": {"+": None, "-": None},
                "revive": {"+": None, "-": None},
                "random": {"+": None, "-": None},
            },
            "arrived_time": {
                "sl": {"+": None, "-": None},
                "ipw": {"+": None, "-": None},
                "revive": {"+": None, "-": None},
                "random": {"+": None, "-": None},
            },
            "get_youjiangsong_reward": {
                "sl": {"+": None, "-": None},
                "ipw": {"+": None, "-": None},
                "revive": {"+": None, "-": None},
                "random": {"+": None, "-": None},
            },
        }

        for predict_type in predict_type_list:
            for exp_type in exp_type_list:  # 正的画一组，负的画一组
                print('-' * 55)
                print(f"city: {city}, predict_type: {predict_type}, exp_type: {exp_type}")
                print('-' * 55)

                for model_type in model_type_list:
                    # load model
                    model = load_model(path_dict[f"{predict_type}_{model_type}_model_path"], data, model_type,
                                       predict_type)

                    # calculate the base point of roi
                    base_pred, base_label = roll_out_model(
                        model, data[f"real_{predict_type}_data"], model_type=model_type, predict_type=predict_type,
                        exp_type=exp_type)

                    # calculate experimental data
                    exp_pred, exp_money, exp_label = roll_out_model(
                        model, data[f"exp_{predict_type}_data"], model_type=model_type, predict_type=predict_type,
                        exp_type=exp_type)

                    # save roi results to results dict
                    money, labels = caculate_roi(base_pred, exp_pred, base_label, exp_label, exp_money, sort=True,
                                                 predict_type=predict_type, exp_type=exp_type)  # 横轴 纵轴
                    random_money, random_labels = caculate_roi(base_pred, exp_pred, base_label, exp_label, exp_money,
                                                               sort=False, predict_type=predict_type, exp_type=exp_type)
                    results[predict_type][model_type][exp_type] = (money, labels)
                    results[predict_type]["random"][exp_type] = (random_money, random_labels)

                money_list = [results[predict_type][k][exp_type][0] for k in results[predict_type].keys()]
                label_list = [results[predict_type][k][exp_type][1] for k in results[predict_type].keys()]
                algo_list = list(results[predict_type].keys())
                print(f"algo list {algo_list}")
                area_list = plot_aucc(money_list, label_list, algo_list, predict_type, exp_type, city)

                for i in range(len(algo_list) - 1):
                    algo = algo_list[i]
                    a = area_list[i]
                    b = area_list[-1]
                    a = a - b
                    aucc = (a + b) / (2 * b)

                    print(f"{algo} {exp_type} aucc: {aucc}")

                    aucc_dict[(city, predict_type, exp_type, algo)] = aucc

    print("\n\n\n")
    print("-" * 55)
    print("汇总aucc")
    for predict_type in predict_type_list:
        for exp_type in exp_type_list:
            for model in model_type_list:
                total_city_aucc = [aucc_dict[((city, predict_type, exp_type, model))] for city in city_list]
                mean = np.mean(total_city_aucc)
                std = np.std(total_city_aucc)
                print(f"[{predict_type} {exp_type} {model}]: {mean} +- {std}")