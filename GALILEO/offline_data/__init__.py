# Created by xionghuichen at 2022/7/26
# Email: chenxh@lamda.nju.edu.cn
# import torch
import gym
import d4rl
import numpy as np
from RLA.easy_log import logger

from tqdm import tqdm
import pickle
import os

LOG_STD_MAX = 2
LOG_STD_MIN = -10

# gen_path = '/opt/meituan/cephfs/user/hadoop-peisongpa/yuzhihua03/mail_sources/d4rl/data/d4rl_gen'


class ExpertDataset:

    def __init__(self, env_name='halfcheetah', train_type='medium', max_length=200, test=False, expand_data=False, generate=None):

        self.env_name = env_name
        self.max_length = max_length
        self.env_types = ['medium', 'expert', 'medium-replay', 'medium-expert']
        # self.env_types = ['medium', 'expert'] # facilitate for debug
        self.train_type = train_type
        self.test = test
        self.expand_data = expand_data
        self.extra_data_trajs = 9000

        # 加载多个环境数据
        self.load_data()
        self.get_info()
        # if generate:             # generate后test是为了做什么呢？
        #     # self.clip_length()
        #     self.generate_data(generate)
        # else:
        #     self.get_info()

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

    def clip_length(self):
        for item in self.env_types:
            for i in range(len(self.data[item]) - 2):
                t1 = self.data[item][i]
                t1 = np.split(t1, np.array(self.data[item][6]).cumsum())[:-1]
                # TODO: currently modified as abandon traj that is shorter than max_length
                t1 = np.concatenate([x[:self.max_length] for x in t1 if x.shape[0] >= self.max_length], axis=0)
                self.data[item][i] = t1

    def traj_gen(self, data):
        max_traj_len = -1
        last_start = 0
        traj_num = 1
        traj_lens = []
        logger.info('[ DEBUG ] obs shape: ', data['observations'].shape)
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
                logger.info('[ DEBUG + WARN ]: trajectory length is too large: current step is ', i, traj_num, traj_len)

        traj_lens.append(data['observations'].shape[0] - last_start)
        assert len(traj_lens) == traj_num
        # assert max_traj_len <= 1000

        # 1, making state and lst action
        last_start_ind = 0
        traj_lens_it = traj_lens  # [traj_num_to_infer * i_ter: min(traj_num_to_infer * (i_ter + 1), traj_num)]
        states = np.zeros((len(traj_lens_it), self.max_length, data['observations'].shape[-1]))
        next_states = np.zeros((len(traj_lens_it), self.max_length, data['next_observations'].shape[-1]))
        actions = np.zeros((len(traj_lens_it), self.max_length, data['actions'].shape[-1]))
        rewards = np.zeros((len(traj_lens_it), self.max_length, 1))
        lst_actions = np.zeros((len(traj_lens_it), self.max_length, data['last_actions'].shape[-1]))
        start_ind = last_start_ind
        for ind, item in enumerate(traj_lens_it):
            target_length = np.minimum(item, self.max_length)
            states[ind, :item] = data['observations'][start_ind:(start_ind + target_length)]
            lst_actions[ind, :target_length] = data['last_actions'][start_ind:(start_ind + target_length)]
            actions[ind, :target_length] = data['actions'][start_ind:(start_ind + target_length)]
            rewards[ind, :target_length] = np.expand_dims(data['rewards'][start_ind:(start_ind + target_length)], axis=-1)
            next_states[ind, :target_length] = data['next_observations'][start_ind:(start_ind + target_length)]
            start_ind += item
        logger.info('[ DEBUG ] size of total env states: {}, actions: {}'.format(states.shape, actions.shape))
        return states, actions, rewards, next_states, lst_actions

    def load_data(self, env_name=None):
        self.data = {}
        self.eval_data = {}
        if not env_name:
            env_name = self.env_name
        for type_it in self.env_types:
            env = gym.make(f'{env_name}-{type_it}-v0')
            dataset = d4rl.qlearning_dataset(env)
            dataset['last_actions'] = np.concatenate((np.zeros((1, dataset['actions'].shape[1])), dataset['actions'][:-1, :]),
                                                  axis=0).copy()
            # logger.info(data['actions'] - data['last_actions'])
            dataset['first_step'] = np.zeros_like(dataset['terminals'])
            dataset['end_step'] = np.zeros_like(dataset['terminals'])
            dataset['valid'] = np.ones_like(dataset['terminals'])
            traj_state, traj_acs, rewards, next_states, _ = self.traj_gen(dataset)
            self.data[type_it] = [traj_state, traj_acs, rewards, next_states]
            traj_num = traj_state.shape[0]
            idx = np.arange(0, traj_num)
            np.random.shuffle(idx)
            idx = idx[:int(traj_num * 0.2)]
            self.eval_data[type_it] = [traj_state[idx], traj_acs[idx], rewards[idx], next_states[idx]]

    # def refactor(self, x, low, high):
    #     return tf.clip_by_value((x - low) / (high - low), 0, 1) * 2 - 1

    def min_max(self, s):
        return np.min(s, axis=(0, 1)), np.max(s, axis=(0, 1))


if __name__ == '__main__':
    dataset = ExpertDataset(env_name='hopper', max_length=200, test=False, expand_data=True)


