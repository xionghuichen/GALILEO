# Created by xionghuichen at 2022/7/26
# Email: chenxh@lamda.nju.edu.cn

from GALILEO.offline_data.dataloader import Dataset
from GALILEO.config import *
from GALILEO.utils import *
from RLA import logger
import numpy as np
from scipy.integrate import romb
from matplotlib import pyplot as plt
from RLA import MatplotlibRecorder as mpr

def correlation(real, pred):
    return np.mean((np.array(pred) - np.mean(pred)) * (np.array(real) - np.mean(real))) / (
                np.array(real).std() * np.array(pred).std())

def correlation_scale(real, pred):
    return np.mean((np.array(pred) - np.mean(pred)) * (np.array(real) - np.mean(real))) / (
                np.square(np.array(real).std()))

class DatasetHandler(object):
    def __init__(self, dataset, data_type, env):
        assert isinstance(dataset, Dataset)
        dataset.gen_data()
        dataset.formulate_data()
        self.dataset = dataset
        self.data_type = data_type
        self.env = env
        self.traj_S = dataset.traj_S
        self.S = dataset.S
        self.traj_A = dataset.traj_A
        self.A = dataset.A
        self.traj_P_S_next = dataset.traj_P_S_next
        self.traj_masks = dataset.traj_masks
        self.P_S_next = dataset.P_S_next
        self.s_mean = dataset.s_min
        self.s_std = dataset.s_max - dataset.s_min
        self.s_std[self.s_std < 1.0] = 1.0
        self.a_mean = dataset.a_min
        self.a_std = (dataset.a_max - dataset.a_min) * 1.2
        self.p_s_n_mean = dataset.ps_min
        self.p_s_n_std = (dataset.ps_max - dataset.ps_min) * 1.2
        self.testA = dataset.testA
        self.testS = dataset.testS
        self.s_dim = len(self.s_mean)
        self.a_dim = len(self.a_mean)
        self.ps_dim = len(self.p_s_n_mean)
        self.data_traj_size = self.traj_S.shape[1]
        self.data_size = self.S.shape[0]

    def evaluation(self, do_plot, predict_fn, dis_pred_fn):
        if self.data_type == DataType.D4RL:
            res_dict = {}
            for item in self.dataset.env_types:
                pred = []
                real = []
                new_testS = flat_traj(self.dataset.eval_data[item][0])
                new_testA = flat_traj(self.dataset.eval_data[item][1])
                new_testR = flat_traj(self.dataset.eval_data[item][2])
                new_testNS = flat_traj(self.dataset.eval_data[item][3])
                new_masks = flat_traj(self.dataset.eval_data[item][5])
                new_testS = new_testS[np.where(new_masks[..., 0] == 1)]
                new_testA = new_testA[np.where(new_masks[..., 0] == 1)]
                new_testR = new_testR[np.where(new_masks[..., 0] == 1)]
                new_testNS = new_testNS[np.where(new_masks[..., 0] == 1)]
                # new_testP_S_next = self.dataset.data['eval'][item][3]
                new_testP_S_next = np.concatenate((new_testR, new_testNS), axis=-1)
                pred.append(predict_fn(new_testS, new_testA))
                real.append(new_testP_S_next)
                pred = np.array(pred)
                real = np.array(real)
                res = np.mean(np.square(np.array(pred) - np.array(real)))
                logger.record_tabular(f"perf/{item}-mse", res)
                res_dict[f"{item}-mse"] = res
            return res_dict
        else:
            if self.data_type == DataType.TCGA:
                numsamples = 64 + 1
            else:
                numsamples = 8 + 1
            stepsize = (self.testA.max() - self.testA.min()) / numsamples
            do_acs = np.arange(self.testA.min(), self.testA.max(), stepsize)

            pred = []
            real = []
            data_pair_pred = []
            data_pair_real = []
            for do_a in do_acs:
                pred_res = predict_fn(self.testS, np.ones(self.testA.shape) * do_a)
                pred.append(pred_res)
                data_pair_pred.append(np.concatenate([self.testS, np.ones(self.testA.shape) * do_a, pred_res], axis=-1))
                real_res = self.env.part_env(self.testS, np.ones(self.testA.shape) * do_a)
                real.append(real_res)
                data_pair_real.append(np.concatenate([self.testS, np.ones(self.testA.shape) * do_a, real_res], axis=-1))
            pred = np.array(pred)
            real = np.array(real)
            mean_pred = np.mean(pred, axis=(1, 2))
            mean_real = np.mean(real, axis=(1, 2))
            logger.record_tabular("perf/do-A-mise", np.mean(romb(np.square(np.array(pred)[..., 0] - np.array(real)[..., 0]), dx=stepsize, axis=0)))
            logger.record_tabular("perf/do-A-mmse", np.mean(np.max(np.square(pred - real), axis=0)))
            logger.record_tabular("perf/do-A-corr", correlation(mean_real, mean_pred))

            delta_range = self.testA.max() - self.testA.min()
            delta_range *= 0.2
            stepsize = delta_range * 2 / numsamples
            do_delta_acs = np.arange(- delta_range, delta_range, stepsize)
            pred = []
            real = []
            data_pair_pred = []
            data_pair_real = []
            for do_a in do_delta_acs:
                pred_res = predict_fn(self.testS, self.testA + do_a)
                pred.append(pred_res)
                data_pair_pred.append(np.concatenate([self.testS, self.testA + do_a, pred_res], axis=-1))
                real_res = self.env.part_env(self.testS, self.testA + do_a)
                real.append(real_res)
                data_pair_real.append(np.concatenate([self.testS, self.testA + do_a, real_res], axis=-1))
            if dis_pred_fn is not None and do_plot:
                data_pair_real = np.asarray(data_pair_real).reshape([-1, data_pair_real[0].shape[-1]])
                data_pair_pred = np.asarray(data_pair_pred).reshape([-1, data_pair_pred[0].shape[-1]])
                res_fake, res_real = dis_pred_fn(
                    data_pair_real[:, :self.s_dim], data_pair_real[:, self.s_dim:self.s_dim+self.a_dim], data_pair_real[:, self.s_dim+self.a_dim:],
                    data_pair_pred[:, :self.s_dim], data_pair_pred[:, self.s_dim:self.s_dim+self.a_dim], data_pair_pred[:, self.s_dim+self.a_dim:])
                res_fake = res_fake.reshape([-1, self.testS.shape[0], 1])
                res_real = res_real.reshape([-1, self.testS.shape[0], 1])

                def plot_fn():
                    plt.plot(do_delta_acs, res_fake.squeeze().mean(1), '--*', label='gen_data_prob')
                    plt.plot(do_delta_acs, res_real.squeeze().mean(1), '--+', label='real_data_prob')
                mpr.pretty_plot_wrapper('dis_pred_delta_acs', plot_fn)

            pred = np.array(pred)
            real = np.array(real)
            mean_pred = np.mean(pred, axis=(1, 2))
            mean_real = np.mean(real, axis=(1, 2))
            logger.record_tabular("perf/do-delta_A-mise", np.mean(romb(np.square(np.array(pred)[..., 0] - np.array(real)[..., 0]), dx=stepsize, axis=0)))
            logger.record_tabular("perf/do-delta-A-corr", correlation(mean_real, mean_pred))            
            logger.record_tabular("perf/do-delta-A-mmse", np.mean(np.max(np.square(pred - real), axis=0)))
            logger.info("do_delta_A mean_pred", mean_pred)
            logger.info("do_delta_A mean_pred", mean_real)
            if do_plot:
                def plot_fn():
                    plt.plot(do_delta_acs, mean_pred, '--*', label='mean_pred')
                    plt.plot(do_delta_acs, mean_real, '--+', label='mean_real')
                mpr.pretty_plot_wrapper('delta_response', plot_fn)
        logger.dump_tabular()
