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
        self.traj_length = None
        self.traj_masks = None
        self.s_max, self.s_min = None, None
        self.a_max, self.a_min = None, None
        self.ps_max, self.ps_min = None, None
        self.horizon = horizon
        self.S = None
        self.A = None
        self.P_S_next = None
        self.testS = None
        self.testA = None

    def gen_data(self, *args, **kwargs):
        pass

    def formulate_data(self, *args, **kwargs):
        pass
    
    def print_info(self):
        logger.info(f"S (min: {self.s_min}, max: {self.s_max})")
        logger.info(f"A (min: {self.a_min}, max: {self.a_max})")
        logger.info(f"PS (min: {self.ps_min}, max: {self.ps_max})")

class D4rlDataset(Dataset):
    def __init__(self, env_name='halfcheetah', train_type='medium', horizon=200, res_next=False):
        super(D4rlDataset, self).__init__(horizon)
        self.env_name = env_name
        self.env_types = ['medium', 'expert', 'medium-replay', 'medium-expert']
        self.train_type = train_type
        self.res_next = res_next

    def get_info(self):
        # 得到state，action 维度
        self.n_state = self.data[self.train_type][0].shape[-1]
        self.n_actions = self.data[self.train_type][1].shape[-1]

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
        last_start_ind = 0
        traj_lens_it = traj_lens  # [traj_num_to_infer * i_ter: min(traj_num_to_infer * (i_ter + 1), traj_num)]
        states = np.zeros((len(traj_lens_it), self.horizon, data['observations'].shape[-1]))
        next_states = np.zeros((len(traj_lens_it), self.horizon, data['next_observations'].shape[-1]))
        actions = np.zeros((len(traj_lens_it), self.horizon, data['actions'].shape[-1]))
        rewards = np.zeros((len(traj_lens_it), self.horizon, 1))
        masks = np.zeros((len(traj_lens_it), self.horizon, 1))
        lst_actions = np.zeros((len(traj_lens_it), self.horizon, data['last_actions'].shape[-1]))
        start_ind = last_start_ind
        clip_length = np.clip(traj_lens_it, 0, self.horizon)
        for ind, item in enumerate(traj_lens_it):
            target_length = np.minimum(item, self.horizon)
            states[ind, :target_length] = data['observations'][start_ind:(start_ind + target_length)]
            lst_actions[ind, :target_length] = data['last_actions'][start_ind:(start_ind + target_length)]
            actions[ind, :target_length] = data['actions'][start_ind:(start_ind + target_length)]
            masks[ind, :target_length] = 1
            rewards[ind, :target_length] = np.expand_dims(data['rewards'][start_ind:(start_ind + target_length)], axis=-1)
            next_states[ind, :target_length] = data['next_observations'][start_ind:(start_ind + target_length)]
            start_ind += item
        print('[ DEBUG ] size of total env states: {}, actions: {}'.format(states.shape, actions.shape))

        return states, actions, rewards, next_states, lst_actions, clip_length, masks

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
            traj_state, traj_acs, rewards, next_states, _, traj_length, masks = self.traj_gen(dataset)
            if self.res_next:
                next_states = next_states - traj_state
            self.data[type_it] = [traj_state, traj_acs, rewards, next_states, traj_length, masks] # [states, actions, next_states, rewards, terminals, start_index, length]
            traj_num = traj_state.shape[0]
            idx = np.arange(0, traj_num)
            np.random.shuffle(idx)
            idx = idx[:int(traj_num * 0.2)]
            self.eval_data[type_it] = [traj_state[idx], traj_acs[idx], rewards[idx], next_states[idx], traj_length[idx], masks[idx]]
            
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
        self.traj_length = dataset[4]
        self.traj_masks = dataset[5].transpose((1, 0, 2))
        self.s_max, self.s_min = self.state_high, self.state_low
        self.a_max, self.a_min = self.action_high, self.action_low
        self.ps_max, self.ps_min = np.concatenate((self.reward_high, self.next_state_high)), \
                                 np.concatenate((self.reward_low, self.next_state_low))
        self.S = self.traj_S[np.where(self.traj_masks[..., 0] == 1)]
        self.A = self.traj_A[np.where(self.traj_masks[..., 0] == 1)]
        # self.masks = np.concatenate(self.traj_masks, axis=0)
        self.P_S_next = self.traj_P_S_next[np.where(self.traj_masks[..., 0] == 1)]
        self.n_actions = self.A.shape[-1]
        self.print_info()


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
        set_global_seeds(self.data_seed)
        s = np.array([np.random.normal(np.ones(self.dim - 1) * i / dataset_size * MAX_S_MEAN, np.ones(self.dim - 1))
                      for i in range(dataset_size)])
        s = np.append(s, np.zeros([s.shape[0], 1]), axis=-1)
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
        S = np.concatenate(S, axis=0)
        A = np.concatenate(A, axis=0)
        P_S_next = np.concatenate(P_S_next, axis=0)
        self.S = S
        self.A = A
        self.P_S_next = P_S_next
        
        testS = S # [valid_test_S]
        testA = A # [valid_test_S]
        logger.info("before valid_test_S computation")
        # logger.info(f"test S: {valid_test_S.shape}, all {S.shape}, ratio {valid_test_S.shape[0] / S.shape[0]}")
        valid_test_A = np.where(np.abs(testA) > 5e-2)[0]
        logger.info(f"test A: {valid_test_A.shape}, all {S.shape}, ratio {valid_test_A.shape[0] / S.shape[0]}")
        testS = S[valid_test_A]
        testA = A[valid_test_A]
        max_test_data_point = 50000
        if testA.shape[0] > max_test_data_point:
            test_idx = np.arange(testA.shape[0])
            np.random.shuffle(test_idx)
            test_idx = test_idx[:max_test_data_point]
            testS = testS[test_idx]
            testA = testA[test_idx]
        self.testS = testS
        self.testA = testA

    def formulate_data(self):
        self.traj_masks = np.ones((self.horizon, self.traj_S.shape[1], 1))
        self.traj_length = np.ones((self.traj_S.shape[0])) * self.horizon
        self.s_max, self.s_min = self.traj_S.max(axis=(0,1)), self.traj_S.min(axis=(0,1))
        self.a_max, self.a_min = self.traj_A.max(axis=(0,1)), self.traj_A.min(axis=(0,1))
        self.ps_max, self.ps_min = self.traj_P_S_next.max(axis=(0,1)), self.traj_P_S_next.min(axis=(0,1))
        self.masks = np.concatenate(self.traj_masks, axis=0)


class TcgaDataset(Dataset):
    def __init__(self, select_treatment, treatment_selection_bias,
                 dosage_selection_bias, data_location, validation_fraction=0.1, test_fraction=0.2):
        super(TcgaDataset, self).__init__(horizon=2)
        dataset_params = dict()
        dataset_params['select_treatment'] = select_treatment
        dataset_params['treatment_selection_bias'] = treatment_selection_bias
        dataset_params['dosage_selection_bias'] = dosage_selection_bias
        dataset_params['validation_fraction'] = validation_fraction
        dataset_params['test_fraction'] = test_fraction
        dataset_params['data_location'] = data_location
        self.dataset_params = dataset_params
        self.data_class = TCGA_Data(self.dataset_params)
    
    def gen_data(self):
        dataset = self.data_class.generate_dataset()
        dataset_train, dataset_val, dataset_test = get_dataset_splits(dataset)
        S = dataset_train['x']
        A = np.expand_dims(dataset_train['d'], axis=-1)
        P_S_next = dataset_train['y']
        self.traj_S = np.repeat(np.expand_dims(S, axis=0), repeats=self.horizon, axis=0)
        self.traj_A = np.repeat(np.expand_dims(A, axis=0), repeats=self.horizon, axis=0)
        self.traj_P_S_next = np.repeat(np.expand_dims(P_S_next, axis=0), repeats=self.horizon, axis=0)
        self.S = self.traj_S.reshape([-1, self.traj_S.shape[-1]])
        self.A = self.traj_A.reshape([-1, self.traj_A.shape[-1]])
        self.P_S_next = self.traj_P_S_next.reshape([-1, self.traj_P_S_next.shape[-1]])
        dataset_size = S.shape[0]
        self.testS = dataset_test['x']
        self.testA = np.expand_dims(dataset_test['d'], axis=-1)
        
        
    def formulate_data(self):
        self.traj_masks = np.ones((self.horizon, self.traj_S.shape[1], 1))
        self.traj_length = np.ones((self.traj_S.shape[0])) * self.horizon
        self.s_max, self.s_min = self.traj_S.max(axis=(0,1)), self.traj_S.min(axis=(0,1))
        self.a_max, self.a_min = self.traj_A.max(axis=(0,1)), self.traj_A.min(axis=(0,1))
        self.ps_max, self.ps_min = self.traj_P_S_next.max(axis=(0,1)), self.traj_P_S_next.min(axis=(0,1))
        self.masks = np.concatenate(self.traj_masks, axis=0)
        self.print_info()
        self.n_actions = 1