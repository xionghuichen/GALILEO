# import torch
import tensorflow as tf
import gym
import d4rl
import numpy as np
from baselines.common import set_global_seeds
from RLA import logger
from GALILEO.envs.gnfc import GnfcEnv, GnfcPolicy
from GALILEO.envs.tcga import TCGA_Data, get_dataset_splits
from GALILEO.config import *

from tqdm import tqdm
import pickle
import os
import tensorflow.contrib.distributions as tfd

LOG_STD_MAX = 2
LOG_STD_MIN = -10


class Dataset(object):
    def __init__(self, horizon):
        self.traj_S = None
        self.traj_A = None
        self.traj_P_S_next = None
        self.s_max, self.s_min = None, None
        self.a_max, self.a_min = None, None
        self.psmax, self.psmin = None, None
        self.horizon = horizon

    def gen_data(self, *args, **kwargs):
        pass

    def formulate_data(self, *args, **kwargs):
        pass

class D4rlDataset(Dataset):
    def __init__(self, env_name='halfcheetah', train_type='medium', horizon=200, res_next=False):
        super(D4rlDataset, self).__init__()
        self.env_name = env_name
        self.horizon = horizon
        self.env_types = ['medium', 'expert', 'medium-replay', 'medium-expert']
        # self.env_types = ['medium', 'expert'] # facilitate for debug
        self.train_type = train_type
        self.res_next = res_next
        self.extra_data_trajs = 9000

    def get_info(self):
        # 得到state，action 维度
        self.n_state = self.data[self.train_type][0].shape[-1]
        self.n_action = self.data[self.train_type][1].shape[-1]

        # 获取state, reward 最大、最小值
        all_state = [self.data[x][0] for x in self.env_types]
        self.state_low, self.state_high = self.min_max(np.concatenate(all_state, axis=0))
        all_action = [self.data[x][1] for x in self.env_types]
        self.action_low, self.action_high = self.min_max(np.concatenate(all_action, axis=0))
        all_reward = [self.data[x][2] for x in self.env_types]
        self.reward_low, self.reward_high = self.min_max(np.concatenate(all_reward, axis=0))
        all_next_state = [self.data[x][3] for x in self.env_types]
        self.next_state_low, self.next_state_high = self.min_max(np.concatenate(all_next_state, axis=0))

    def traj_gen(self, data):
        max_traj_len = -1
        last_start = 0
        traj_num = 1
        traj_lens = []
        print('[ DEBUG ] obs shape: ', data['observations'].shape)
        for i in range(data['observations'].shape[0]):
            non_terminal = True
            if i >= 1:
                non_terminal = (data['observations'][i] == data['next_observations'][i - 1]).all()
                if data['terminals'][i - 1]:
                    non_terminal = False

            traj_len = i - last_start
            if not non_terminal:
                if data['terminals'][i - 1]:
                    data['next_observations'][i - 1] = data['observations'][i - 1]
                data['last_actions'][i] = 0
                data['first_step'][i] = 1
                data['end_step'][i - 1] = 1

                max_traj_len = max(max_traj_len, traj_len)
                last_start = i
                traj_num += 1
                traj_lens.append(traj_len)

            if traj_len > 1001:
                print('[ DEBUG + WARN ]: trajectory length is too large: current step is ', i, traj_num, traj_len)

        traj_lens.append(data['observations'].shape[0] - last_start)
        assert len(traj_lens) == traj_num
        # assert max_traj_len <= 1000

        # 1, making state and lst action
        last_start_ind = 0
        traj_lens_it = traj_lens  # [traj_num_to_infer * i_ter: min(traj_num_to_infer * (i_ter + 1), traj_num)]
        states = np.zeros((len(traj_lens_it), self.horizon, data['observations'].shape[-1]))
        next_states = np.zeros((len(traj_lens_it), self.horizon, data['next_observations'].shape[-1]))
        actions = np.zeros((len(traj_lens_it), self.horizon, data['actions'].shape[-1]))
        rewards = np.zeros((len(traj_lens_it), self.horizon, 1))
        lst_actions = np.zeros((len(traj_lens_it), self.horizon, data['last_actions'].shape[-1]))
        start_ind = last_start_ind
        for ind, item in enumerate(traj_lens_it):
            target_length = np.minimum(item, self.horizon)
            states[ind, :item] = data['observations'][start_ind:(start_ind + target_length)]
            lst_actions[ind, :target_length] = data['last_actions'][start_ind:(start_ind + target_length)]
            actions[ind, :target_length] = data['actions'][start_ind:(start_ind + target_length)]
            rewards[ind, :target_length] = np.expand_dims(data['rewards'][start_ind:(start_ind + target_length)], axis=-1)
            next_states[ind, :target_length] = data['next_observations'][start_ind:(start_ind + target_length)]
            start_ind += item
        print('[ DEBUG ] size of total env states: {}, actions: {}'.format(states.shape, actions.shape))
        return states, actions, rewards, next_states, lst_actions

    def gen_data(self, env_name=None):
        self.data = {}
        self.eval_data = {}
        if not env_name:
            env_name = self.env_name
        for type_it in self.env_types:
            env = gym.make(f'{env_name}-{type_it}-v0')
            dataset = d4rl.qlearning_dataset(env)
            dataset['last_actions'] = np.concatenate((np.zeros((1, dataset['actions'].shape[1])), dataset['actions'][:-1, :]),
                                                  axis=0).copy()
            # print(data['actions'] - data['last_actions'])
            dataset['first_step'] = np.zeros_like(dataset['terminals'])
            dataset['end_step'] = np.zeros_like(dataset['terminals'])
            dataset['valid'] = np.ones_like(dataset['terminals'])
            traj_state, traj_acs, rewards, next_states, _ = self.traj_gen(dataset)
            if self.res_next:
                next_states = next_states - traj_state
            self.data[type_it] = [traj_state, traj_acs, rewards, next_states] # [states, actions, next_states, rewards, terminals, start_index, length]
            traj_num = traj_state.shape[0]
            idx = np.arange(0, traj_num)
            np.random.shuffle(idx)
            idx = idx[:int(traj_num * 0.2)]
            self.eval_data[type_it] = [traj_state[idx], traj_acs[idx], rewards[idx], next_states[idx]]

    def refactor(self, x, low, high):
        return tf.clip_by_value((x - low) / (high - low), 0, 1) * 2 - 1

    def min_max(self, s):
        return np.min(s, axis=(0, 1)), np.max(s, axis=(0, 1))

    def formulate_data(self):
        self.get_info()
        dataset = self.data[self.train_type]
        self.traj_S = dataset[0].transpose((1, 0, 2))
        self.traj_A = dataset[1].transpose((1, 0, 2))
        self.traj_P_S_next = np.concatenate((dataset[2], dataset[3]), axis=-1).transpose((1, 0, 2))
        self.s_max, self.s_min = self.state_high, self.state_low
        self.a_max, self.a_min = self.action_high, self.action_low
        self.psmax, self.psmin = np.concatenate((self.reward_high, self.state_high)), \
                                 np.concatenate((self.reward_low, self.state_low))


class GnfcDataset(Dataset):
    def __init__(self, dim, dm_noise, one_step_dyn, target_line, random_prob, noise_scale, horizon, data_seed, dataset_traj_size):
        super(GnfcDataset, self).__init__(horizon)
        self.dataset_traj_size = dataset_traj_size
        self.env = GnfcEnv(dim, dm_noise, one_step_dyn, target_line)
        self.policy = GnfcPolicy(target_line, random_prob, noise_scale)
        self.dim = dim
        self.data_seed = data_seed

    def gen_data(self):
        dataset_size = max(int(max(self.dim * 100, 1000) / self.horizon), self.dataset_traj_size)
        S = []
        A = []
        P_S_next = []
        s = np.array([np.random.normal(np.ones(self.dim - 1) * i / dataset_size * MAX_S_MEAN, np.ones(self.dim - 1))
                      for i in range(dataset_size)])
        s = np.append(s, np.zeros([s.shape[0], 1]), axis=-1)

        set_global_seeds(self.data_seed)
        for i in range(self.horizon):
            a = self.policy.act(s)
            next_s, r, done, p_s_next = self.env.step(s, a)
            S.append(s)
            A.append(a)
            P_S_next.append(p_s_next)
            s = next_s
        self.traj_S = np.asarray(S)
        self.traj_A = np.asarray(A)
        self.traj_P_S_next = np.asarray(P_S_next)

    def formulate_dataset(self):
        self.s_max, self.s_min = self.traj_S.max(axis=(0,1)), self.traj_S.min(axis=(0,1))
        self.a_max, self.a_min = self.traj_A.max(axis=(0,1)), self.traj_A.min(axis=(0,1))
        self.psmax, self.psmax = self.traj_P_S_next.max(axis=(0,1)), self.traj_P_S_next.min(axis=(0,1))


class TcgnDataset(Dataset):
    def __init__(self, select_treatment, treatment_selection_bias,
                 dosage_selection_bias, validation_fraction=0.1, test_fraction=0.2):
        super(TcgnDataset, self).__init__(horizon=2)
        dataset_params = dict()
        dataset_params['select_treatment'] = select_treatment
        dataset_params['treatment_selection_bias'] = treatment_selection_bias
        dataset_params['dosage_selection_bias'] = dosage_selection_bias
        dataset_params['validation_fraction'] = validation_fraction
        dataset_params['test_fraction'] = test_fraction

        data_class = TCGA_Data(dataset_params)
        dataset = data_class.dataset
        dataset_train, dataset_val, dataset_test = get_dataset_splits(dataset)
        S = dataset_train['x']
        A = np.expand_dims(dataset_train['d'], axis=-1)
        P_S_next = dataset_train['y']
        self.traj_S = np.repeat(np.expand_dims(S, axis=0), repeats=self.horizon, axis=0)
        self.traj_A = np.repeat(np.expand_dims(A, axis=0), repeats=self.horizon, axis=0)
        self.traj_P_S_next = np.repeat(np.expand_dims(P_S_next, axis=0), repeats=self.horizon, axis=0)
        dataset_size = S.shape[0]
        testS = dataset_test['x']
        testA = np.expand_dims(dataset_test['d'], axis=-1)

    def formulate_dataset(self):
        self.s_max, self.s_min = self.traj_S.max(axis=(0,1)), self.traj_S.min(axis=(0,1))
        self.a_max, self.a_min = self.traj_A.max(axis=(0,1)), self.traj_A.min(axis=(0,1))
        self.psmax, self.psmax = self.traj_P_S_next.max(axis=(0,1)), self.traj_P_S_next.min(axis=(0,1))
