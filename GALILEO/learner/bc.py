import numpy as np

from GALILEO.nets.net import *
from GALILEO.learner.base import BaseLearner
from GALILEO.losses.base import *
from GALILEO.utils import *
from GALILEO.config import *
import tensorflow as tf

class BC(BaseLearner):
    def __init__(self, dm_model, opt, lr, dataset_holder, sess):
        super(BC, self).__init__(dataset_holder)
        self.dm_model = dm_model
        self.opt = opt
        self.lr = lr
        self.sess = sess

    def graph_construction(self, scope='bc', reuse=False):
        assert isinstance(self.dm_model, DM)
        with tf.variable_scope(scope, reuse=reuse):
            norm_s_input = self.norm_s(self.s_input_ph)
            norm_a_input = self.norm_a(self.a_input_ph)
            norm_p_s_next_target = self.norm_p_s_next(self.p_s_next_target_ph)
            self.dist, self.dm_mean, self.dm_std = self.dm_model.obj_graph_construct((norm_s_input, norm_a_input))
            self.dm_sample = self.dist.sample()
            self.loss = tf.reduce_mean(neg_loglikelihoood(self.dist, norm_p_s_next_target))
            for v in self.dm_model.trainable_variables():
                print("train", v)
            self.train_op = self.opt(learning_rate=self.lr, name='sl_opt').minimize(self.loss, var_list=self.dm_model.trainable_variables())

    def evaluate_next_state(self, s, a, deter=True):
        if deter:
            norm_ps_next = self.sess.run(self.dm_mean, feed_dict={self.s_input_ph: s, self.a_input_ph: a})
        else:
            norm_ps_next = self.sess.run(self.dm_sample, feed_dict={self.s_input_ph: s, self.a_input_ph: a})
        return self.denorm_p_s_next(norm_ps_next)
