# Created by xionghuichen at 2022/8/25
# Email: chenxh@lamda.nju.edu.cn

from GALILEO.config import *

d4rl_config = {
    'common': {
        'horizon': 1000,
        'atten_dis': True,
        'iters': 50000,
        'hid_dim': 512,
        'max_kl': 0.001,
        "occ_noise_coef": 4.0,
        "auto_d_noise": False,
        'std_bound': 0.02,
        'dis_noise': 0.1,
        
    },
    'walker2d': {
        'common': {
        },
        'medium': {
            'common' :{
            },
            '100': {
                'occ_noise_coef': 1.0
            }
        },
    },
    'halfcheetah': {
        'common': {
        },
        'medium-replay': {
            'common': {
                'occ_noise_coef': 2.0
            },
        },
    },
    'hopper': {
        'common': {
            'std_bound': 0.1,
            'max_kl': 0.0005
        },
    }
}


tcga_config = {
    'common': {
        'atten_dis': False, # atten_dis seems to be better, but we have not test the results in this setting
        'gamma': 0.0,
        'horizon': 2,
        'lr_dm': 1e-5,
        'iters': 10000,
        'occ_noise_coef': 0.0,
    }
}

gnfc_config = {
    'common': {
        'atten_dis': False,  # atten_dis seems to be better, but we have not test the results in this setting
        'lr_dm': 1e-5,
        'iters': 50000,
        'occ_noise_coef': 0.1,
        'std_bound': 0.005,
        # reduce computation cost.
        'sample_traj_size': 100,
        'dis_batch_traj_size': 100,
        'bc_batch_size': 10000,
        'bc_step': 1,
    }
}

sl_config = {
    # the std variable of neural network in the standard SL is reduced slowly, which is much faster in GALILEO.
    # for fair comparisons, we use the ground-truth std as the initial value of neural network in SL algorithm.
    DataType.GNFC: {
        'dm_std_init': 0.02,
    },
    DataType.D4RL: {
        'dm_std_init': 0.02,
    },
}

dm_1000_model_map = {
        'hopper': {
            'galileo' : [
                '2023/03/16/02-05-10-529196',
                ],
        },
        'walker2d': {
            'galileo' : [
                '2023/05/02/03-47-02-213431',
                # '2023/04/30/00-32-34-582116',
                # '2023/05/01/14-49-51-214924'
                '2023/04/30/00-35-03-504979'
                # '2023/03/16/22-03-53-138255',
                # '2023/03/16/09-53-23-873972',
                ],
        },
        'halfcheetah': {
            'galileo' : [
                '2023/03/16/02-06-01-414782',
                # '2023/03/16/17-59-29-964035',
            ]
        }

}
dm_model_map = {
    'sl': {
        'halfcheetah': {
            'loaded_task_name': 'd4rl-v4',
            'loaded_date': '2022/10/25/16-27-41-349935',
        },
        'hopper': {
            'loaded_task_name': 'd4rl-v4',
            'loaded_date': '2022/10/25/16-23-23-018318',
        },
        'walker2d': {
            'loaded_task_name': 'd4rl-v4',
            'loaded_date': '2022/10/25/16-20-03-039793',
        }
    },
    'ipw': {
        'halfcheetah': {
            'loaded_task_name': 'd4rl-v4',
            'loaded_date': '2022/11/10/13-53-53-408106',
        },
        'hopper': {
            'loaded_task_name': 'd4rl-v4',
            'loaded_date': '2022/11/10/13-52-09-900949',
        },
        'walker2d': {
            'loaded_task_name': 'd4rl-v4',
            'loaded_date': '2022/11/10/13-47-13-549010',
        }
    },
    'galileo': {
        'halfcheetah': {
            'loaded_task_name': 'd4rl-v4',
            'loaded_date': '2022/10/11/21-59-45-471232',
            # 'loaded_date': '2022/10/04/17-00-25-041857',
            },
        'hopper': {
            'loaded_task_name': 'd4rl-v4',
            'loaded_date': '2022/10/11/14-43-10-014624',
        },
        'walker2d': {
            'loaded_task_name': 'd4rl-v4',
            'loaded_date': '2022/10/13/19-17-07-264598',
        }
    },
    'ganite': {
        'halfcheetah': {
            'loaded_task_name': 'd4rl-v4',
            'loaded_date': '2022/11/09/03-12-55-216604',
            # 'loaded_date': '2022/10/04/17-00-25-041857',
        },
        'hopper': {
            'loaded_task_name': 'd4rl-v4',
            'loaded_date': '2022/11/09/03-09-35-200190',
        },
        'walker2d': {
            'loaded_task_name': 'd4rl-v4',
            'loaded_date': '2022/11/09/03-06-14-667989',
        }
    }
}
dm_model_map[AlgType.REAL_ENV_MODEL_ROLLOUT] = dm_model_map[AlgType.GALILEO]

def update_hp(kwargs):
    if kwargs['data_type'] == DataType.D4RL:
        kwargs.update(d4rl_config['common'])
        if kwargs['env_name'] in d4rl_config.keys():
            kwargs.update(d4rl_config[kwargs['env_name']]['common'])
            if kwargs['data_train_type'] in d4rl_config[kwargs['env_name']].keys():
                kwargs.update(d4rl_config[kwargs['env_name']][kwargs['data_train_type']]['common'])
                if str(kwargs['horizon']) in d4rl_config[kwargs['env_name']][kwargs['data_train_type']].keys():
                    kwargs.update(d4rl_config[kwargs['env_name']][kwargs['data_train_type']][str(kwargs['horizon'])])

    elif kwargs['data_type'] == DataType.TCGA:
        kwargs.update(tcga_config['common'])
        kwargs['env_name'] = 'tcga'
    elif kwargs['data_type'] == DataType.GNFC:
        kwargs.update(gnfc_config['common'])
        kwargs['env_name'] = 'gnfc'
    else:
        raise NotImplementedError
    if kwargs['alg_type'] == AlgType.SL:
        kwargs.update(sl_config[kwargs['data_type']])
    if kwargs['auto_d_noise']:
        kwargs['dis_noise'] = kwargs['std_bound']
    if kwargs['alg_type'] == AlgType.GAIL:
        kwargs['dis_noise'] = 0.0
        kwargs['only_model_likelihood'] = True
        kwargs['rescale_grad'] = False
    if kwargs['alg_type'] == AlgType.GANITE:
        kwargs['dis_noise'] = 0.0
        kwargs['bc_step'] = 0
        kwargs['rescale_grad'] = False
        kwargs['gamma'] = 0.0