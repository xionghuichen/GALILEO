# Created by xionghuichen at 2022/10/25
# Email: chenxh@lamda.nju.edu.cn
import os
import gym
from stable_baselines.common.vec_env import DummyVecEnv
from GALILEO.sac2.sac import SAC
from GALILEO.sac2 import policies
from RLA import exp_manager, ExperimentLoader
from stable_baselines.common.base_class import _UnvecWrapper
import argparse
from baselines.common.misc_util import (
    set_global_seeds,
    boolean_flag,
)
import tensorflow as tf


from GALILEO.utils import *
from GALILEO.config import *
from auto_config_map import *
from GALILEO.offline_data.dataset_handler import DatasetHandler
from GALILEO.offline_data.dataloader import *
from GALILEO.losses.base import *
from GALILEO.envs.term_fn import is_terminal
from GALILEO.learner.dynamics_model import DMEnv
from GALILEO.nets.net import DM
from GALILEO.envs.mujoco import MjEnv

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--env_name', type=str, default='halfcheetah')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--info', type=str, default='van_sac')
    parser.add_argument('--alg_type', type=str, default=AlgType.GALILEO)
    boolean_flag(parser, 'render', default=False)
    parser.add_argument('--buffer_size', type=int, default=1000000)
    parser.add_argument('--layers', type=int, default=2)
    parser.add_argument('--layer_size', type=int, default=256)
    parser.add_argument('--sac_batch_size', type=int, default=256)
    parser.add_argument('--sac_learning_starts', type=int, default=10000)
    parser.add_argument('--episode_len', type=int, default=20)
    parser.add_argument('--gamma', type=int, default=0.99)
    parser.add_argument('--reg_weight', type=float, default=0.0)  # 0.0
    parser.add_argument('--target_entropy', type=str, default='auto') # 0.0
    parser.add_argument('--target_entropy_coef', type=float, default=3.0)  # 0.0
    parser.add_argument('--rew_scale', type=float, default=1.0)  # 0.0
    parser.add_argument('--acs_cons_scale', type=float, default=2)  # 0.0
    parser.add_argument('--sac_ent_coef', type=str, default='auto_0.2')  # auto_0.2
    parser.add_argument('--total_timesteps', type=int, default=400000)
    boolean_flag(parser, 'uniform_std', default=False)
    boolean_flag(parser, 'state_cons', default=True)
    boolean_flag(parser, 'deter_pred', default=False)
    boolean_flag(parser, 'use_tfp_imp', default=True)
    boolean_flag(parser, 'concate_ac', default=True)
    boolean_flag(parser, 'branch_init', default=False)
    boolean_flag(parser, 'use_xavier_init', default=True)
    boolean_flag(parser, 'real_reset', default=False)


    args = parser.parse_args()
    args.buffer_size = int(args.buffer_size / 1000 * 100)
    args.sac_learning_starts = int(args.sac_learning_starts / 1000 * 100)
    if args.episode_len < 100:
        args.gamma = 0.9
    # elif args.episode_len < 10:
    #     args.gamma = 0.5
    # if args.env_name == 'halfcheetah':
    #     args.total_timesteps = 2000000
    kwargs = vars(args)
    # kwargs.update(dense_config_dict[args.env_name])
    exp_manager.set_hyper_param(**kwargs)
    # add parameters to track:
    exp_manager.add_record_param(['info', 'seed', 'episode_len', 'env_name', 'alg_type',
                                  'deter_pred', 'acs_cons_scale', 'real_reset'])
    return kwargs

if __name__ == '__main__':
    kwargs = parse_args()
    seed = kwargs['seed']
    set_global_seeds(seed)
    if kwargs['use_xavier_init']:
        policies.default_initializer = tf.contrib.layers.xavier_initializer()
    else:
        policies.default_initializer = tf.compat.v1.keras.initializers.he_normal()
    args = argparse.Namespace(**kwargs)


    # env = gym.make(kwargs['env_id'])
    # env = DummyVecEnv([lambda: env])
    # env = normalize(GymEnv(kwargs['env_id']))
    def env_creator(env_id, unwarper):
        explr_env = gym.make(env_id).env
        # for rllab env
        # explr_env = normalize(GymEnv(env_id))
        # explr_env.metadata = None
        explr_env = DummyVecEnv([lambda: explr_env])
        if unwarper:
            explr_env = _UnvecWrapper(explr_env)
        return explr_env

    if args.env_name == 'halfcheetah':
        generate = 'HalfCheetah-v2'
    elif args.env_name == 'walker2d':
        generate = 'Walker2d-v2'
    elif args.env_name == 'hopper':
        generate = 'Hopper-v2'
    else:
        raise NotImplementedError

    env = env_creator(generate, False)
    explr_env = env_creator(generate, True)
    eval_env = env_creator(generate, True)


    def get_package_path():
        return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    exp_manager.configure(task_table_name='default-v2',
                          rla_config=os.path.join(get_package_path(), 'rla_config.yaml'),
                          ignore_file_path=os.path.join(get_package_path(), '.gitignore'),
                          data_root=get_package_path())
    exp_manager.log_files_gen()
    exp_manager.print_args()

    # construct layer param:
    layer_param = [kwargs['layer_size'] for i in range(kwargs['layers'])]
    policy_kwargs = {
        "layers": layer_param,
        "reg_weight": kwargs['reg_weight'],
        "uniform_std": kwargs['uniform_std'],
        "use_tfp_imp": kwargs['use_tfp_imp'],
        "concate_ac": kwargs['concate_ac'],
    }
    # TODO:
    #  1. rew_scale = kwargs['rew_scale']
    #  2. episode_len=kwargs['episode_len']

    model = SAC(policy=policies.MlpPolicy, env=env, buffer_size=kwargs['buffer_size'],
                gamma=kwargs['gamma'],  seed=kwargs['seed'],
                verbose=1, batch_size=kwargs['sac_batch_size'], ent_coef=kwargs['sac_ent_coef'],
                target_entropy=kwargs['target_entropy'], episode_len=kwargs['episode_len'],
                learning_starts=kwargs['sac_learning_starts'], policy_kwargs=policy_kwargs,
                eval_env=eval_env, target_entropy_coef=kwargs['target_entropy_coef'])
    # init dynamics model
    sess = model.sess  # U.make_session(make_default=True)
    if args.alg_type == AlgType.REAL_ENV:
        dm_env = None
    else:
        exp_loader = ExperimentLoader()
        args.loaded_task_name = dm_model_map[args.alg_type][args.env_name]['loaded_task_name']
        args.loaded_date = dm_model_map[args.alg_type][args.env_name]['loaded_date']
        exp_loader.config(task_name=args.loaded_task_name, record_date=args.loaded_date, root=get_package_path())
        model_args = exp_loader.import_hyper_parameters()
        set_global_seeds(model_args.data_seed)
        dataset = D4rlDataset(env_name=args.env_name, horizon=model_args.horizon)
        env = gym.make(generate)
        env = MjEnv(env)
        data_handler = DatasetHandler(dataset=dataset, data_type=model_args.data_type, env=env)
        terminal_fn = lambda s, a, ns: is_terminal(s, a, ns, args.env_name)
        n_actions = data_handler.a_dim
        dim_next = len(data_handler.p_s_n_mean)
        with model.graph.as_default():
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
                dm_env = DMEnv(data_handler=data_handler, dm_model=dm_model, sess=sess, terminal_fn=terminal_fn,
                               gym_env=env.env, branch_init=args.branch_init, episode_len=kwargs['episode_len'],
                               deter_pred=kwargs['deter_pred'], use_real_env=args.alg_type == AlgType.REAL_ENV_MODEL_ROLLOUT,
                               real_reset=kwargs['real_reset'], acs_cons_scale=args.acs_cons_scale, state_cons=args.state_cons)
                dm_env.graph_construction()
                sess.run(tf.initialize_all_variables())
                exp_manager.new_saver(max_to_keep=1)
                exp_loader.load_from_record_date(var_prefix=dm_model.scope)
                logger.record_tabular("perf/std", np.mean(sess.run(dm_env.dm_std)))
        res_dict = data_handler.evaluation(do_plot=True, predict_fn=dm_env.evaluate_next_state, dis_pred_fn=None)
        dm_env.mean_pred_error = res_dict['medium-mse']
        set_global_seeds(args.seed)
    model.learn(total_timesteps=args.total_timesteps, log_interval=5, dm_env=dm_env)