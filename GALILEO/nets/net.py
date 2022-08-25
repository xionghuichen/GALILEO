import tensorflow as tf
import tensorflow.contrib.distributions as tfd

LOG_STD_MAX = 2
LOG_STD_MIN = -10

def DM_net(s, a, dim_next, reuse, std_bound, name='dm_net'): # generate P_S_next aka reward
    dm_input = tf.concat([s, a], axis=-1)
    with tf.variable_scope(name, reuse=reuse):
        # policy_input = tf.reduce_mean(policy_input, axis=-1, keepdims=True)
        x_h = tf.layers.dense(dm_input, units=512, activation=tf.nn.leaky_relu)
        x_h = tf.layers.dense(x_h, units=512, activation=tf.nn.leaky_relu)
        x_h = tf.layers.dense(x_h, units=512, activation=tf.nn.leaky_relu)
        x_h = tf.layers.dense(x_h, units=512, activation=tf.nn.leaky_relu)
        # y = tf.layers.dense(x_h, units=1)
        # mean = tf.sigmoid(tf.layers.dense(x_h, units=dim_next)/5) * 1.1
        mean = tf.nn.tanh(tf.layers.dense(x_h, units=dim_next)) * 1.2
        # std = tf.exp(tf.get_variable(name="std", shape=[1, 1], initializer=tf.zeros_initializer())) * 0.3 + 1e-6
        # std = ((tf.tanh(tf.layers.dense(x_h, units=1)) + 1) * 0.3 + 1e-4)
        logstd = tf.get_variable(name="std", shape=[dim_next], initializer=tf.constant_initializer(-1.3))
        logstd = tf.clip_by_value(logstd, LOG_STD_MIN, LOG_STD_MAX)
        # std = tf.exp(logstd) * 0.3 + 1e-6
        std = tf.exp(logstd)
    return tfd.Normal(loc=mean, scale=std)


def dis_net(s, a, s_next, reuse, name='dis_net'):
    dm_input = tf.concat([s, a, s_next], axis=-1)
    input_dim = dm_input.shape[-1].value
    with tf.variable_scope(name, reuse=reuse):
        # policy_input = tf.reduce_mean(policy_input, axis=-1, keepdims=True)
        x_h = tf.layers.dense(dm_input, units=512, activation=tf.nn.leaky_relu)
        x_h = tf.layers.dense(x_h, units=512, activation=tf.nn.leaky_relu)
        x_h = tf.layers.dense(x_h, units=512, activation=tf.nn.leaky_relu)
        x_h = tf.layers.dense(x_h, units=512, activation=tf.nn.leaky_relu)
        x_h = tf.layers.dense(x_h, units=512, activation=tf.nn.leaky_relu)
        residual = tf.layers.dense(x_h, units=input_dim)
        x = dm_input + residual
        x = tf.nn.tanh(x)
        y = tf.layers.dense(x, units=512, activation=tf.nn.leaky_relu)
        y = tf.layers.dense(y, units=512, activation=tf.nn.leaky_relu)
        y = tf.layers.dense(y, units=512, activation=tf.nn.leaky_relu)
        y = tf.layers.dense(y, units=512, activation=tf.nn.leaky_relu)
        y = tf.layers.dense(y, units=512, activation=tf.nn.leaky_relu)
        y = tf.layers.dense(y, units=1)
        
    return y

def v_net(s, reuse, name='v_net'):

    with tf.variable_scope(name, reuse=reuse):
        # policy_input = tf.reduce_mean(policy_input, axis=-1, keepdims=True)
        x_h = tf.layers.dense(s, units=512, activation=tf.nn.leaky_relu)
        x_h = tf.layers.dense(x_h, units=512, activation=tf.nn.leaky_relu)
        x_h = tf.layers.dense(x_h, units=512, activation=tf.nn.leaky_relu)
        x_h = tf.layers.dense(x_h, units=512, activation=tf.nn.leaky_relu)
        x_h = tf.layers.dense(x_h, units=512, activation=tf.nn.leaky_relu)
        y = tf.layers.dense(x_h, units=1)
    return y


def pi_net(s, reuse, std_bound, n_actions, init_std=0.3, name='pi_net'):
    input = s
    with tf.variable_scope(name, reuse=reuse):
        # policy_input = tf.reduce_mean(policy_input, axis=-1, keepdims=True)
        x_h = tf.layers.dense(input, units=512, activation=tf.nn.leaky_relu)
        x_h = tf.layers.dense(x_h, units=512, activation=tf.nn.leaky_relu)
        x_h = tf.layers.dense(x_h, units=512, activation=tf.nn.leaky_relu)
        x_h = tf.layers.dense(x_h, units=512, activation=tf.nn.leaky_relu)
        # mean = tf.layers.dense(x_h, units=1)
        # mean = tf.sigmoid(tf.layers.dense(x_h, units=n_actions) / 5) * 1.1
        mean = tf.nn.tanh(tf.layers.dense(x_h, units=n_actions)) * 1.2

        # std = tf.exp(tf.get_variable(name="std", shape=[1, 1], initializer=tf.zeros_initializer())) + 1e-3
        # std = (tf.tanh(tf.layers.dense(x_h, units=1)) + 1) * 0.3 + std_bound
        logstd = tf.get_variable(name="std", shape=[n_actions], initializer=tf.constant_initializer(-1.3))
        # logstd = tf.clip_by_value(logstd, tf.log(std_bound/init_std), LOG_STD_MAX)
        logstd = tf.clip_by_value(logstd, LOG_STD_MIN, LOG_STD_MAX)
        # std = tf.exp(logstd) * init_std
        std = tf.exp(logstd)
    return tfd.Normal(loc=mean, scale=std)