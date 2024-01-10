import tensorflow as tf
import tensorflow.contrib.distributions as tfd
import numpy as np
from GALILEO.nets.base import  TfBasicClass
from GALILEO.utils import *
LOG_STD_MAX = 2
LOG_STD_MIN = -10

def layer_block(block_input, units, activation=tf.nn.leaky_relu, rate=0.1, training=False, use_atten_block=False):
    x_h = tf.layers.dense(block_input, units=units, activation=activation)
    if use_atten_block:
        x_h = tf.layers.dropout(x_h, rate=rate, training=training)
        x_h = block_input + x_h
        x_h = tf.contrib.layers.layer_norm(x_h)
    return x_h


class DM(TfBasicClass):
    
    def __init__(self, hid_dim, dim_next, dm_std_init=0.3, scope='dm_net'):
        super(DM, self).__init__(scope)
        self.dim_next = dim_next
        self.hid_dim = hid_dim
        self.dm_std_init = dm_std_init

    def _obj_construct(self, source_input):
        s, a = source_input
        dm_input = tf.concat([s, a], axis=-1)
        x_h = tf.layers.dense(dm_input, units=self.hid_dim, activation=tf.nn.leaky_relu)
        x_h = tf.layers.dense(x_h, units=self.hid_dim, activation=tf.nn.leaky_relu)
        x_h = tf.layers.dense(x_h, units=self.hid_dim, activation=tf.nn.leaky_relu)
        x_h = tf.layers.dense(x_h, units=self.hid_dim, activation=tf.nn.leaky_relu)
        # x_h = tf.layers.dense(x_h, units=self.hid_dim, activation=tf.nn.leaky_relu)
        mean = (tf.nn.tanh(tf.layers.dense(x_h, units=self.dim_next)) + 0.8) / 1.6
        # mean = tf.sigmoid(tf.layers.dense(x_h, units=self.dim_next)/5) * 1.1

        logstd = tf.get_variable(name="std", shape=[self.dim_next], initializer=tf.zeros_initializer())
        logstd = tf.clip_by_value(logstd, LOG_STD_MIN, LOG_STD_MAX)
        std = tf.exp(logstd) * self.dm_std_init + 1e-6
        return tfd.Normal(loc=mean, scale=std), mean, std
    
class Discriminator(TfBasicClass):
    
    def __init__(self, hid_dim, dis_noise, mask_s_next, s_add_noise, res_dis, occ_noise_coef, scope='dis_net'):
        super(Discriminator, self).__init__(scope)
        self.hid_dim = hid_dim
        self.dis_noise = dis_noise
        self.s_add_noise = s_add_noise
        self.mask_s_next = mask_s_next
        self.res_dis = res_dis
        self.occ_noise_coef = occ_noise_coef
    
    def _obj_construct(self, source_input):
        s, a, s_next = source_input
        if self.mask_s_next:
            s_next *= 0
        dis_input = tf.concat([s, a, s_next], axis=-1)
        input_dim = dis_input.shape[-1].value
        if not self.res_dis:
            x_h = tf.layers.dense(dis_input, units=self.hid_dim, activation=tf.nn.leaky_relu)
            x_h = tf.layers.dense(x_h, units=self.hid_dim, activation=tf.nn.leaky_relu)
            x_h = tf.layers.dense(x_h, units=self.hid_dim, activation=tf.nn.leaky_relu)
            x_h = tf.layers.dense(x_h, units=self.hid_dim, activation=tf.nn.leaky_relu)
            x_h = tf.layers.dense(x_h, units=self.hid_dim, activation=tf.nn.leaky_relu)
            x_h = tf.layers.dense(x_h, units=self.hid_dim, activation=tf.nn.leaky_relu)
            y = tf.layers.dense(x_h, units=1)
        else:
            x_h = tf.layers.dense(dis_input, units=self.hid_dim, activation=tf.nn.leaky_relu)
            x_h = tf.layers.dense(x_h, units=self.hid_dim, activation=tf.nn.leaky_relu)
            x_h = tf.layers.dense(x_h, units=self.hid_dim, activation=tf.nn.leaky_relu)
            x_h = tf.layers.dense(x_h, units=self.hid_dim, activation=tf.nn.leaky_relu)
            x_h = tf.layers.dense(x_h, units=self.hid_dim, activation=tf.nn.leaky_relu)
            residual = tf.layers.dense(x_h, units=input_dim)
            x = dis_input + residual
            x = tf.nn.tanh(x)
            y = tf.layers.dense(x, units=self.hid_dim, activation=tf.nn.leaky_relu)
            y = tf.layers.dense(y, units=self.hid_dim, activation=tf.nn.leaky_relu)
            y = tf.layers.dense(y, units=self.hid_dim, activation=tf.nn.leaky_relu)
            y = tf.layers.dense(y, units=self.hid_dim, activation=tf.nn.leaky_relu)
            y = tf.layers.dense(y, units=self.hid_dim, activation=tf.nn.leaky_relu)
            y = tf.layers.dense(y, units=1)
        return y
    
    def logits(self, source_input, with_noise):
        s, a, s_next = source_input
        if with_noise:
            if self.s_add_noise:
                return self.obj_graph_construct((s + tf.random_normal(tf.shape(s), 0, self.dis_noise * self.occ_noise_coef),
                                                 a + tf.random_normal(tf.shape(a), 0, self.dis_noise),
                                                 s_next + tf.random_normal(tf.shape(s_next), 0, self.dis_noise)))
            else:
                return self.obj_graph_construct((s, a + tf.random_normal(tf.shape(a), 0, self.dis_noise),
                                                 s_next + tf.random_normal(tf.shape(s_next), 0, self.dis_noise)))
        else:
            return self.obj_graph_construct((s, a, s_next))
    
    def prob(self, source_input, with_noise):
        return tf.nn.sigmoid(self.logits(source_input, with_noise))

    def reward(self, source_input, with_noise):
        return - tf.log(1 - self.prob(source_input, with_noise) + 1e-8)


class V(TfBasicClass):
    
    def __init__(self, hid_dim, scope='v_net'):
        super(V, self).__init__(scope)
        self.hid_dim = hid_dim
        
    def _obj_construct(self, source_input):
        s = source_input
        x_h = tf.layers.dense(s, units=self.hid_dim, activation=tf.nn.leaky_relu)
        x_h = tf.layers.dense(x_h, units=self.hid_dim, activation=tf.nn.leaky_relu)
        x_h = tf.layers.dense(x_h, units=self.hid_dim, activation=tf.nn.leaky_relu)
        x_h = tf.layers.dense(x_h, units=self.hid_dim, activation=tf.nn.leaky_relu)
        x_h = tf.layers.dense(x_h, units=self.hid_dim, activation=tf.nn.leaky_relu)
        # x_h = tf.layers.dense(x_h, units=self.hid_dim, activation=tf.nn.leaky_relu)
        y = tf.layers.dense(x_h, units=1)
        return y
    

class Pi(TfBasicClass):
    
    def __init__(self, std_bound, n_actions, hid_dim, init_std=0.3, scope='pi_net'):
        super(Pi, self).__init__(scope)
        self.hid_dim = hid_dim
        self.std_bound = std_bound
        self.init_std = init_std
        self.n_actions = n_actions
    
    def _obj_construct(self, source_input):
        s = source_input
        x_h = tf.layers.dense(s, units=self.hid_dim, activation=tf.nn.leaky_relu)
        x_h = tf.layers.dense(x_h, units=self.hid_dim, activation=tf.nn.leaky_relu)
        x_h = tf.layers.dense(x_h, units=self.hid_dim, activation=tf.nn.leaky_relu)
        x_h = tf.layers.dense(x_h, units=self.hid_dim, activation=tf.nn.leaky_relu)
        x_h = tf.layers.dense(x_h, units=self.hid_dim, activation=tf.nn.leaky_relu)
        mean = (tf.nn.tanh(tf.layers.dense(x_h, units=self.n_actions)) + 0.8) / 1.6
        # mean = tf.sigmoid(tf.layers.dense(x_h, units=self.n_actions) / 5) * 1.1
        logstd = tf.get_variable(name="std", shape=[self.n_actions], initializer=tf.constant_initializer(np.log(self.init_std)))
        logstd = tf.clip_by_value(logstd, tf.log(self.std_bound), LOG_STD_MAX)
        std = tf.exp(logstd)
        return tfd.Normal(loc=mean, scale=std), mean, std
