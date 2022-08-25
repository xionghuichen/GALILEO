# Created by xionghuichen at 2022/7/26
# Email: chenxh@lamda.nju.edu.cn

from RLA.easy_log.tester import exp_manager
from RLA.easy_log import logger
from RLA.easy_log.simple_mat_plot import simple_plot, simple_hist
from RLA.rla_argparser import arg_parser_postprocess, boolean_flag
import argparse
import os
from GALILEO.utils import *
from GALILEO.config import *
from auto_config_map import *
from GALILEO.offline_data.dataset_handler import DatasetHandler
from GALILEO.offline_data.dataloader import *



def argsparser():
    parser = argparse.ArgumentParser("Train coupon policy in simulator")
    parser.add_argument('--seed',  type=int, default=8)
    parser.add_argument('--data_seed',  type=int, default=9)
    parser.add_argument("--data_type", default=DataType.D4RL, type=str)
    parser.add_argument("--horizon", default=50, type=int)
    parser.add_argument("--iters", default=50000, type=int)
    parser.add_argument("--g_step", default=3, type=int)
    # for d4rl
    parser.add_argument("--task_name", default='walker2d', type=str)
    # for tcga
    parser.add_argument("--select_treatment", default=2, type=int)
    parser.add_argument("--treatment_selection_bias", default=2.0, type=float)
    parser.add_argument("--dosage_selection_bias", default=2.0, type=float)
    # for gnfc
    parser.add_argument('--dim',  type=int, default=2)
    parser.add_argument('--alpha',  type=float, default=0.8)
    parser.add_argument('--dm_noise',  type=float, default=2.0)
    parser.add_argument('--noise_scale',  type=float, default=0.2)
    parser.add_argument('--random_prob',  type=float, default=0.05)
    parser.add_argument('--data_seed',  type=int, default=888)
    boolean_flag(parser, 'one_step_dyn', default=False)
    # for rla
    parser.add_argument('--info', default='default exp info', type=str)
    parser.add_argument('--loaded_date', default='', type=str)
    parser.add_argument('--loaded_task_name', default='', type=str)

    args = parser.parse_args()
    return args

def main():
    args = argsparser()

    def get_package_path():
        return os.path.dirname(os.path.abspath(__file__))
    kwargs = vars(args)
    # hyper-parameter map
    if args.data_type == DataType.D4RL:
        kwargs.update(d4rl_config['common'])
        kwargs.update(d4rl_config[args.env_name])
    elif args.data_type == DataType.TCGA:
        kwargs.update(tcga_config['common'])
    elif args.data_type == DataType.GNFC:
        kwargs.update(gnfc_config['common'])
    else:
        raise NotImplementedError
    # rla config
    exp_manager.set_hyper_param(**kwargs)
    exp_manager.add_record_param(['info', 'seed', 'data_type', 'env_name', 'horizon','std_bound'])
    exp_manager.configure(task_table_name=args.data_type + '-v3',
                     rla_config=os.path.join(get_package_path(), 'rla_config.yaml'),
                     ignore_file_path=os.path.join(get_package_path(), '.gitignore'),
                     data_root=get_package_path())
    exp_manager.log_files_gen()
    exp_manager.print_args()
    # dataset generation
    set_global_seeds(args.data_seed)
    if args.data_type == DataType.D4RL:
        dataset = D4rlDataset(env_name=args.env_name, horizon=args.horizon)
    elif args.data_type == DataType.TCGA:
        dataset = TcgnDataset(args.select_treatment, args.treatment_selection_bias, args.dosage_selection_bias)
    elif args.data_type == DataType.GNFC:
        target_line = MAX_S_MEAN / args.alpha
        dataset = GnfcDataset(args.dim, args.dm_noise, args.one_step_dyn, target_line,
                              args.random_prob, args.noise_scale, args.horizon, args.data_seed, dataset_traj_size=100000)
    else:
        raise NotImplementedError
    data_handler = DatasetHandler(dataset=dataset)
    exp_manager.new_saver(max_to_keep=1)
    for i in range(args.iters):
        for gi in range(args.g_step):
            S_start = data_handler.traj_S[0]



