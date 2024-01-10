# Created by xionghuichen at 2022/7/26
# Email: chenxh@lamda.nju.edu.cn


import numpy as np
import random
import tensorflow as tf


def set_global_seeds(i):
    try:
        import MPI
        rank = MPI.COMM_WORLD.Get_rank()
    except ImportError:
        rank = 0

    myseed = i + 1000 * rank if i is not None else None
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


def zipsame(*seqs):
    L = len(seqs[0])
    assert all(len(seq) == L for seq in seqs[1:])
    return zip(*seqs)

def var_shape(x):
    out = x.get_shape().as_list()
    assert all(isinstance(a, int) for a in out), \
        "shape function assumes that shape is fully known"
    return out

def numel(x):
    return intprod(var_shape(x))

def intprod(x):
    return int(np.prod(x))


def flatgrad(loss, var_list, clip_norm=None):
    grads = tf.gradients(loss, var_list)
    if clip_norm is not None:
        grads = [tf.clip_by_norm(grad, clip_norm=clip_norm) for grad in grads]
    return tf.concat(axis=0, values=[
        tf.reshape(grad if grad is not None else tf.zeros_like(v), [numel(v)])
        for (v, grad) in zip(var_list, grads)
    ])


def set_from_flat(var_list, dtype=tf.float32):
    shapes = list(map(var_shape, var_list))
    total_size = np.sum([intprod(shape) for shape in shapes])

    theta = tf.placeholder(dtype, [total_size])
    start = 0
    assigns = []
    for (shape, v) in zip(shapes, var_list):
        size = intprod(shape)
        assigns.append(tf.assign(v, tf.reshape(theta[start:start + size], shape)))
        start += size
    return tf.group(*assigns), theta


def get_flat(var_list):
    return tf.concat(axis=0, values=[tf.reshape(v, [numel(v)]) for v in var_list])


def soft_clip(input, min_v, max_v):
    output = max_v - tf.nn.softplus(max_v - input)
    output = min_v + tf.nn.softplus(output - min_v)
    # output = tf.clip_by_value(input, min_v, max_v)
    return output


# TODO: use jax
def cg(f_Ax, b, cg_iters=10, callback=None, verbose=False, residual_tol=1e-10):
    """
    Demmel p 312
    """
    p = b.copy()
    r = b.copy()
    x = np.zeros_like(b)
    rdotr = r.dot(r)

    fmtstr =  "%10i %10.3g %10.3g"
    titlestr =  "%10s %10s %10s"
    if verbose: print(titlestr % ("iter", "residual norm", "soln norm"))

    for i in range(cg_iters):
        if callback is not None:
            callback(x)
        if verbose: print(fmtstr % (i, rdotr, np.linalg.norm(x)))
        z = f_Ax(p)
        v = rdotr / p.dot(z)
        x += v*p
        r -= v*z
        newrdotr = r.dot(r)
        mu = newrdotr/rdotr
        p = r + mu*p

        rdotr = newrdotr
        if rdotr < residual_tol:
            break

    if callback is not None:
        callback(x)
    if verbose: print(fmtstr % (i+1, rdotr, np.linalg.norm(x)))  # pylint: disable=W0631
    return x

