# Created by xionghuichen at 2023/4/28
# Email: chenxh@lamda.nju.edu.cn

import argparse
import gym
import os
import os.path as osp
import numpy as np
from GALILEO.evaluation.dope_policy import D4RLPolicy
from GALILEO.evaluation.func import *
from auto_config_map import dm_1000_model_map
from RLA import exp_manager, ExperimentLoader
import random

from GALILEO.offline_data.dataset_handler import DatasetHandler
from GALILEO.offline_data.dataloader import *
from GALILEO.losses.base import *
from GALILEO.envs.term_fn import is_terminal
from GALILEO.learner.dynamics_model import DMEnv
from GALILEO.nets.net import DM
from GALILEO.envs.mujoco import MjEnv

from baselines.common import tf_util as U

def set_global_seeds(i):
    try:
        import MPI
        rank = MPI.COMM_WORLD.Get_rank()
    except ImportError:
        rank = 0

    myseed = i  + 1000 * rank if i is not None else None
    try:
        import tensorflow as tf
        tf.set_random_seed(myseed)
    except ImportError:
        pass
    np.random.seed(myseed)
    random.seed(myseed)

def argsparser():
    parser = argparse.ArgumentParser("Train coupon policy in simulator")
    parser.add_argument('--seed', type=int, default=6)
    parser.add_argument('--info', type=str, default='')
    parser.add_argument('--load_eval_policy_path', type=str, default='../data/dope_policy/')
    parser.add_argument('--env_name', type=str, default='halfcheetah')
    parser.add_argument('--alg_type', type=str, default='galileo')
    parser.add_argument("--num-eval-episodes", type=int, default=10)
    args = parser.parse_args()
    return args


def get_package_path():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


if __name__ == '__main__':
    args = argsparser()

    kwargs = vars(args)
    dope_name = args.env_name
    if args.env_name == 'halfcheetah':
        generate = 'HalfCheetah-v2'
    elif args.env_name == 'walker2d':
        generate = 'Walker2d-v2'
        dope_name =  'walker'
    elif args.env_name == 'hopper':
        generate = 'Hopper-v2'
    else:
        raise NotImplementedError

    real_env = gym.make(generate)
    args.obs_shape = real_env.observation_space.shape
    args.action_dim = np.prod(real_env.action_space.shape)
    args.max_action = real_env.action_space.high[0]
    eval_policy_set = [D4RLPolicy(osp.join(args.load_eval_policy_path, args.env_name, f"{dope_name}_online_{idx}.pkl"), device='cuda:0') for idx in range(10)]

    real_values = None
    data_handler = None
    exp_loader = ExperimentLoader()
    args.loaded_task_name = 'd4rl-v4'
    datas = dm_1000_model_map[args.env_name][args.alg_type]

    if real_values is None:
        real_values = []
        for idx, policy in enumerate(eval_policy_set):
            print(f"eval_idx {idx}")
            real_value = compute_real_value(real_env, policy, num_eval_episodes=args.num_eval_episodes)
            real_values.append(real_value)
    real_values = np.array(real_values)
    value_min, value_max = real_values.min(), real_values.max()
    print("value_min", value_min, "value_max", value_max)

    for data in []:
        print("eval", args.alg_type, data)
        args.loaded_date = data
        exp_loader.config(task_name=args.loaded_task_name, record_date=args.loaded_date, root=get_package_path())
        model_args = exp_loader.import_hyper_parameters()
        set_global_seeds(model_args.data_seed)
        dataset = D4rlDataset(env_name=args.env_name, horizon=model_args.horizon)
        env = real_env
        env = MjEnv(env)
        if data_handler is None:
            data_handler = DatasetHandler(dataset=dataset, data_type=model_args.data_type, env=env)
        terminal_fn = lambda s, a, ns: is_terminal(s, a, ns, args.env_name)
        n_actions = data_handler.a_dim
        dim_next = len(data_handler.p_s_n_mean)
        sess = U.make_session(make_default=True)

        with sess.as_default():
            if args.alg_type == AlgType.GALILEO or args.alg_type == AlgType.REAL_ENV_MODEL_ROLLOUT or args.alg_type == AlgType.GANITE:
                prefix = AlgType.GALILEO
            elif args.alg_type == AlgType.SL:
                prefix = 'bc'
            elif args.alg_type == AlgType.IPW:
                prefix = 'ipw'
            if not hasattr(model_args, 'dm_std_init'):
                model_args.dm_std_init = 0.3  # for code compatibility
            # model_args.dm_std_init *= 20
            dm_model = DM(model_args.hid_dim, dim_next, model_args.dm_std_init, scope=prefix + '/dm_net')
            dm_env = DMEnv(dataset_holder=data_handler, dm_model=dm_model, sess=sess, terminal_fn=terminal_fn,
                           gym_env=env.env, branch_init=False, episode_len=1000,
                           deter_pred=True, use_real_env=args.alg_type == AlgType.REAL_ENV_MODEL_ROLLOUT,
                           real_reset=True, acs_cons_scale=-1, state_cons=False)
            dm_env.graph_construction()
            sess.run(tf.initialize_all_variables())
            exp_loader.load_from_record_date(var_prefix=dm_model.scope)
            print("perf/std", np.mean(sess.run(dm_env.dm_std)))


        fake_values = []
        for idx, policy in enumerate(eval_policy_set):
            print(f"eval_idx {idx}")
            fake_value = compute_real_value(dm_env, policy, num_eval_episodes=args.num_eval_episodes)
            fake_values.append(fake_value)

        real_values, fake_values = np.array(real_values), np.array(fake_values)
        value_min, value_max = real_values.min(), real_values.max()
        norm_real_values = (real_values - value_min) / (value_max - value_min)
        norm_fake_values = (fake_values - value_min) / (value_max - value_min)

        print(norm_real_values)
        print(norm_fake_values)
        absolute_error = (np.abs(norm_real_values - norm_fake_values)).mean()
        raw_absolute_error = (np.abs(real_values - fake_values)).mean()
        rank_correlation = np.corrcoef(norm_real_values, norm_fake_values)[0, 1]
        top_idxs = np.argsort(norm_fake_values)[-1:]
        regret = norm_real_values.max() - norm_real_values[top_idxs].max()
        print(f"absolute error: {absolute_error}")
        print(f"raw absolute error: {raw_absolute_error}")
        print(f"rank correlation: {rank_correlation}")
        print(f"regret: {regret}")



