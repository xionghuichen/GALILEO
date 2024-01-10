import tensorflow as tf
import numpy as np


def neg_loglikelihoood(dist, target, clip_v=10):
    return - tf.minimum(tf.log(dist.prob(target) + 1e-8), clip_v)


def logsigmoid(a):
    '''Equivalent to tf.log(tf.sigmoid(a))'''
    return -tf.nn.softplus(-a)

def logit_bernoulli_entropy(logits):
    """
      - S(l) log(S(l)) - (1 - S(l)) log( (1 - S(l)) )
    = - 1/(1+e^{-l}) log(1/(1+e^{-l})) - (e^{-l}/(1+e^{-l})) log(e^{-l}/(1+e^{-l}))
    = - 1/(1+e^{-l}) log(1/(1+e^{-l})) - \
        (e^{-l}/(1+e^{-l})) log(e^{-l}} - (e^{-l}/(1+e^{-l})) log(1/(1+e^{-l}))
    = - (e^{-l}/(1+e^{-l})) log(e^{-l}} - log(1/(1+e^{-l}))
    = (1. - tf.nn.sigmoid(logits)) * logits - logsigmoid(logits)
    :param logits:
    :return:
    """
    ent = (1. - tf.nn.sigmoid(logits)) * logits - logsigmoid(logits)
    return ent


def add_vtarg_and_adv(seg, gamma, lam, rollout_step, traj_size, adv_type):
    if adv_type == 'sas':
        postfix = '_SAS'
        target_v_postfix = 'V_SAS'
    elif adv_type == 'sa':
        postfix = '_SA'
        target_v_postfix = 'V_SA'
    elif adv_type == 'mixed':
        postfix = '_SAS'
        target_v_postfix = 'V_SA'
        # seg["adv_mixed"] = seg['tdlamret_sas'] - seg['tdlamret_sa']
        # return
    else:
        raise NotImplementedError
    new = np.concatenate([np.asarray(seg["D"]), np.zeros([1, np.asarray(seg["D"]).shape[1], 1])], axis=0)
    vpred = np.concatenate(
        [np.asarray(seg[target_v_postfix]), np.expand_dims(np.asarray(seg["V_next"]), axis=0)], axis=0)

    rew = seg["R" + postfix]
    seg["adv_" + adv_type] = gaelam = np.empty([rollout_step, traj_size, 2], 'float32')
    lastgaelam = np.zeros([traj_size, 2], 'float32')
    for t in reversed(range(rollout_step)):
        nonterminal = 1 - new[t + 1]
        delta = rew[t] + gamma * vpred[t + 1] * nonterminal - vpred[t]
        gaelam[t] = lastgaelam = delta + gamma * lam * nonterminal * lastgaelam
    seg["tdlamret_" + adv_type] = seg["adv_" + adv_type] + seg["V" + postfix]


def remove_masked_data(seg, have_done):
    to_remove = np.array(have_done).astype(np.bool)[..., 0]
    for k in seg.keys():
        seg[k] = np.asarray(seg[k])
        if len(seg[k].shape) == 3:
            seg[k] = seg[k][np.where(~to_remove)]
        else:
            seg[k] = np.asarray(seg[k]).reshape([-1, seg[k].shape[-1]])
