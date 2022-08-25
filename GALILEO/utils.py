# Created by xionghuichen at 2022/7/26
# Email: chenxh@lamda.nju.edu.cn


import numpy as np
import random
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


def flat_traj(traj_input):
    if len(traj_input.shape) == 3:
        return traj_input.reshape([-1, traj_input.shape[-1]])
    elif len(traj_input.shape) == 2:
        return traj_input.reshape([-1])
    else:
        raise NotImplementedError