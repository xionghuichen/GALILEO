# Created by xionghuichen at 2022/11/7
# Email: chenxh@lamda.nju.edu.cn

from GALILEO.nets.net import *
from GALILEO.learner.base import BaseLearner
from GALILEO.losses.base import *
from GALILEO.utils import *
from GALILEO.config import *
import tensorflow as tf


class IPW(BaseLearner):
    def __init__(self, dm_model, pi_model, opt, pi_opt, lr, other_lr, dataset_holder, sess):
        super(IPW, self).__init__(dataset_holder)
        self.dm_model = dm_model
        self.pi_model = pi_model
        self.opt = opt
        self.pi_opt = pi_opt
        self.lr = lr
        self.other_lr = other_lr
        self.sess = sess

    def graph_construction(self, scope='ipw', reuse=False):
        assert isinstance(self.dm_model, DM)
        with tf.variable_scope(scope, reuse=reuse):
            norm_s_input = self.norm_s(self.s_input_ph)
            norm_a_input = self.norm_a(self.a_input_ph)
            norm_p_s_next_target = self.norm_p_s_next(self.p_s_next_target_ph)
            self.dist, self.dm_mean, self.dm_std = self.dm_model.obj_graph_construct((norm_s_input, norm_a_input))
            self.dm_sample = self.dist.sample()
            self.pi_dist, self.pi_mean, self.pi_std = self.pi_model.obj_graph_construct((norm_s_input))
            self.pi_likelihood_loss = tf.reduce_mean(neg_loglikelihoood(self.pi_dist, norm_a_input))
            policy_opt = self.pi_opt(learning_rate=self.other_lr, name='pi_opt')
            self.pi_likelihood_op = policy_opt.minimize(self.pi_likelihood_loss, var_list=self.pi_model.trainable_variables())

            self.ipw_loss = tf.reduce_mean(tf.stop_gradient(tf.reduce_mean(tf.clip_by_value(1/(self.pi_dist.prob(norm_a_input)+1e-3), 0.05, 20), axis=-1, keepdims=True)) *
                                       neg_loglikelihoood(self.dist, norm_p_s_next_target))
            self.mse_loss = tf.reduce_mean(neg_loglikelihoood(self.dist, norm_p_s_next_target))
            for v in self.dm_model.trainable_variables():
                print("train", v)
            self.train_op = self.opt(learning_rate=self.lr, name='sl_opt').minimize(self.ipw_loss, var_list=self.dm_model.trainable_variables())

    def evaluate_next_state(self, s, a, deter=True):
        if deter:
            norm_ps_next = self.sess.run(self.dm_mean, feed_dict={self.s_input_ph: s, self.a_input_ph: a})
        else:
            norm_ps_next = self.sess.run(self.dm_sample, feed_dict={self.s_input_ph: s, self.a_input_ph: a})
        return self.denorm_p_s_next(norm_ps_next)

