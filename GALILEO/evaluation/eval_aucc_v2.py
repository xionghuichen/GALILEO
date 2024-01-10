# Created by xionghuichen at 2023/6/7
# Email: chenxh@lamda.nju.edu.cn
import numpy as np
import os
import matplotlib.pyplot as plt

def plot_aucc(ax, money_list, label_list, algo_list, predict_type, exp_type):
    # makedirs
    # if save_path is None:
    #     save_path = f"figs/{city}"
    #     if not os.path.exists(save_path):
    #         os.makedirs(save_path)
    # save_path = os.path.join(save_path, f"{predict_type}_{exp_type}.png")

    area_list = []
    for i in range(len(algo_list)):
        x, y = np.abs(money_list[i]), label_list[i]
        # x, y = np.cumsum(x), np.cumsum(y)
        algo = algo_list[i]
        if algo == "random":
            x = np.array([0, x[-1]])
            y = np.array([0, y[-1]])
        else:
            x = np.hstack([0, x])
            y = np.hstack([0, y])
        ax.plot(x, y, '--*', label=algo)

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

    ax.set_xlabel(f"{exp_type} money value.")
    ax.ticklabel_format(style='sci', scilimits=(-1, 2), axis='both')
    ax.set_ylabel(predict_type)
    ax.legend()
    # fig.savefig(save_path)
    return area_list

# 计算样本的关于预测值pred的roi，并按照roi排序，返回排序后的money及
quantile_list = np.linspace(0, 1, 5)[1:]

def caculate_roi_v2(es_uplift, cost, labels, control_or_treat, exp_type):
    list_to_sort = list(zip(es_uplift, cost, labels, control_or_treat))
    if exp_type == '+':
        list_to_sort.sort(key=lambda x: x[0], reverse=False) # 加钱时间减小，为了让面积增大，需要让时间下降最快的放前面
    elif exp_type == '-':
        list_to_sort.sort(key=lambda x: x[0], reverse=True) # 减钱时间增加，为了让面积增大，需要让时间增加最快的放前面
    list_to_sort = np.array(list_to_sort)
    exp_start, base_start = 0, 0
    sorted_money = list_to_sort[:, 1]
    sorted_label = list_to_sort[:, 2]
    sorted_types = list_to_sort[:, 3]
    money_list = np.zeros(len(quantile_list))
    y_list = np.zeros(len(quantile_list))
    for i in range(len(quantile_list)):
        exp_indices = np.arange(0, int(quantile_list[i] * len(sorted_money)))
        cur_labels = sorted_label[exp_indices]
        cur_types = sorted_types[exp_indices]
        cur_money = sorted_money[exp_indices]
        if np.sum(cur_types) == 0 or np.sum((1 - cur_types)) == 0:
            continue
        # 3. 计算每一个采样分桶（exp_indices）下的随机实验（实际作用的是扰动值）的反馈和真实流量（实际作用的是基础值）的反馈的gap,
        # 作为该分桶下的uplift的评估。如果es_uplift越准确，实际uplift越高的值（不论是随机实验还是真实流量）就会靠前, 此时的y_list才会更大.
        # money_list[i] = (np.sum(cur_money * cur_types) / np.sum(cur_types) - np.sum(cur_money * (1 - cur_types)) / np.sum((1 - cur_types)))
        # y_list[i] = np.sum(cur_labels * cur_types) / np.sum(cur_types) - np.sum(cur_labels * (1 - cur_types)) / np.sum((1 - cur_types))
        # money_list[i] *= exp_indices.shape[0]
        # y_list[i] *=  exp_indices.shape[0]
        # 4. 数据有限时，这部分结果不稳定，把AUCC换成，给定treatment预算下的预估收益
        y_list[i] = np.sum(cur_labels * cur_types) / np.sum(cur_types) - np.sum(cur_labels * (1 - cur_types)) / np.sum((1 - cur_types))
        money_list[i] = np.sum(cur_money * cur_types)
        # exp_start = int(quantile_list[i] * len(sorted_money))
    return money_list, y_list


def caculate_roi(base_pred, exp_pred, base_label, exp_label, money, sort=True,  exp_type="+"):
    # base pred: 对照组使用model的预测值
    # exp_pred: 实验组使用model的预测值
    # label: 对照组和实验组真实的label（对应于pred）
    # money：实验组的实验money
    # return: money, delta labels (sorted)
    # roi = (exp_pred - base_pred) / (delta money)

    base_pred_mean = base_pred.mean(0)
    base_label_mean = base_label.mean(0)

    roi = (exp_pred - base_pred_mean) / (abs(money) + 1e-6)

    if exp_type == "-":
        roi = - roi

    list_to_sort = list(zip(roi, money, exp_label))
    if sort:
        list_to_sort.sort(key=lambda x: x[0], reverse=True)  # x: roi, money, exp_label

    list_to_sort = np.array(list_to_sort)

    exp_start, base_start = 0, 0
    money = list_to_sort[:, 1]
    exp_label = list_to_sort[:, 2]  # TODO: base label需要排序吗

    money_list = np.zeros(len(quantile_list)) # 这不是真实的花销。
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