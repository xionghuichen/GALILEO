# Created by xionghuichen at 2022/10/25
# Email: chenxh@lamda.nju.edu.cn
import sys
import time
from collections import deque
import warnings
import numpy as np
import tensorflow as tf
import os

from stable_baselines.common import tf_util, OffPolicyRLModel, SetVerbosity, TensorboardWriter
from stable_baselines.common.vec_env import VecEnv
import random
from stable_baselines.ppo2.ppo2 import safe_mean, get_schedule_fn
from stable_baselines.sac.policies import SACPolicy
from RLA import logger
from RLA import exp_manager
from RLA import ImgRecorder


class ReplayBuffer(object):
    def __init__(self, size):
        """
        Implements a ring buffer (FIFO).

        :param size: (int)  Max number of transitions to store in the buffer. When the buffer overflows the old
            memories are dropped.
        """
        self._storage = []
        self._maxsize = size
        self._next_idx = 0

    def __len__(self):
        return len(self._storage)

    @property
    def storage(self):
        """[(np.ndarray, float, float, np.ndarray, bool)]: content of the replay buffer"""
        return self._storage

    @property
    def buffer_size(self):
        """float: Max capacity of the buffer"""
        return self._maxsize

    def can_sample(self, n_samples):
        """
        Check if n_samples samples can be sampled
        from the buffer.

        :param n_samples: (int)
        :return: (bool)
        """
        return len(self) >= n_samples

    def is_full(self):
        """
        Check whether the replay buffer is full or not.

        :return: (bool)
        """
        return len(self) == self.buffer_size

    def add(self, obs_t, action, reward, obs_tp1, done):
        """
        add a new transition to the buffer

        :param obs_t: (Any) the last observation
        :param action: ([float]) the action
        :param reward: (float) the reward of the transition
        :param obs_tp1: (Any) the current observation
        :param done: (bool) is the episode done
        """
        data = (obs_t, action, reward, obs_tp1, done)

        if self._next_idx >= len(self._storage):
            self._storage.append(data)
        else:
            self._storage[self._next_idx] = data
        self._next_idx = (self._next_idx + 1) % self._maxsize

    def _encode_sample(self, idxes):
        obses_t, actions, rewards, obses_tp1, dones = [], [], [], [], []
        for i in idxes:
            data = self._storage[i]
            obs_t, action, reward, obs_tp1, done = data
            obses_t.append(np.array(obs_t, copy=False))
            actions.append(np.array(action, copy=False))
            rewards.append(reward)
            obses_tp1.append(np.array(obs_tp1, copy=False))
            dones.append(done)
        return np.array(obses_t), np.array(actions), np.array(rewards), np.array(obses_tp1), np.array(dones)

    def sample(self, batch_size, **_kwargs):
        """
        Sample a batch of experiences.

        :param batch_size: (int) How many transitions to sample.
        :return:
            - obs_batch: (np.ndarray) batch of observations
            - act_batch: (numpy float) batch of actions executed given obs_batch
            - rew_batch: (numpy float) rewards received as results of executing act_batch
            - next_obs_batch: (np.ndarray) next set of observations seen after executing act_batch
            - done_mask: (numpy bool) done_mask[i] = 1 if executing act_batch[i] resulted in the end of an episode
                and 0 otherwise.
        """
        idxes = [random.randint(0, len(self._storage) - 1) for _ in range(batch_size)]
        return self._encode_sample(idxes)
    #
    # def sample_recent(self, batch_size, recent_num=5000):





def calculate_episode_reward_logger(rew_acc, rewards, masks, steps, name='episode_reward'):
    """
    calculates the cumulated episode reward, and prints to tensorflow log the output

    :param rew_acc: (np.array float) the total running reward
    :param rewards: (np.array float) the rewards
    :param masks: (np.array bool) the end of episodes
    :param steps: (int) the current timestep
    :return: (np.array float) the updated total running reward
    :return: (np.array float) the updated total running reward
    """
    with tf.variable_scope("environment_info", reuse=True):
        for env_idx in range(rewards.shape[0]):
            dones_idx = np.sort(np.argwhere(masks[env_idx]))
            if len(dones_idx) == 0:
                rew_acc[env_idx] += sum(rewards[env_idx])
            else:
                rew_acc[env_idx] += sum(rewards[env_idx, :dones_idx[0, 0]])
                exp_manager.time_step_holder.set_time(steps + dones_idx[0, 0])
                logger.record_tabular('episode/' + name, rew_acc[env_idx])

                for k in range(1, len(dones_idx[:, 0])):
                    rew_acc[env_idx] = sum(rewards[env_idx, dones_idx[k-1, 0]:dones_idx[k, 0]])
                    exp_manager.time_step_holder.set_time(steps + dones_idx[k, 0])
                    logger.record_tabular('episode/' + name, rew_acc[env_idx])
                logger.dump_tabular()
                # there may have not terminal state at the end of batch data,
                # so we pushing the reward to the array and calculating at next times.
                rew_acc[env_idx] = sum(rewards[env_idx, dones_idx[-1, 0]:])
    return rew_acc

def get_vars(scope):
    """
    Alias for get_trainable_vars

    :param scope: (str)
    :return: [tf Variable]
    """
    return tf_util.get_trainable_vars(scope)


class SAC(OffPolicyRLModel):
    """
    Soft Actor-Critic (SAC)
    Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor,
    This implementation borrows code from original implementation (https://github.com/haarnoja/sac)
    from OpenAI Spinning Up (https://github.com/openai/spinningup) and from the Softlearning repo
    (https://github.com/rail-berkeley/softlearning/)
    Paper: https://arxiv.org/abs/1801.01290
    Introduction to SAC: https://spinningup.openai.com/en/latest/algorithms/sac.html

    :param policy: (SACPolicy or str) The policy model to use (MlpPolicy, CnnPolicy, LnMlpPolicy, ...)
    :param env: (Gym environment or str) The environment to learn from (if registered in Gym, can be str)
    :param gamma: (float) the discount factor
    :param learning_rate: (float or callable) learning rate for adam optimizer,
        the same learning rate will be used for all networks (Q-Values, Actor and Value function)
        it can be a function of the current progress (from 1 to 0)
    :param buffer_size: (int) size of the replay buffer
    :param batch_size: (int) Minibatch size for each gradient update
    :param tau: (float) the soft update coefficient ("polyak update", between 0 and 1)
    :param ent_coef: (str or float) Entropy regularization coefficient. (Equivalent to
        inverse of reward scale in the original SAC paper.)  Controlling exploration/exploitation trade-off.
        Set it to 'auto' to learn it automatically (and 'auto_0.1' for using 0.1 as initial value)
    :param train_freq: (int) Update the model every `train_freq` steps.
    :param learning_starts: (int) how many steps of the model to collect transitions for before learning starts
    :param target_update_interval: (int) update the target network every `target_network_update_freq` steps.
    :param gradient_steps: (int) How many gradient update after each step
    :param target_entropy: (str or float) target entropy when learning ent_coef (ent_coef = 'auto')
    :param action_noise: (ActionNoise) the action noise type (None by default), this can help
        for hard exploration problem. Cf DDPG for the different action noise type.
    :param random_exploration: (float) Probability of taking a random action (as in an epsilon-greedy strategy)
        This is not needed for SAC normally but can help exploring when using HER + SAC.
        This hack was present in the original OpenAI Baselines repo (DDPG + HER)
    :param verbose: (int) the verbosity level: 0 none, 1 training information, 2 tensorflow debug
    :param tensorboard_log: (str) the log location for tensorboard (if None, no logging)
    :param _init_setup_model: (bool) Whether or not to build the network at the creation of the instance
    :param policy_kwargs: (dict) additional arguments to be passed to the policy on creation
    :param full_tensorboard_log: (bool) enable additional logging when using tensorboard
        Note: this has no effect on SAC logging for now
    :param seed: (int) Seed for the pseudo-random generators (python, numpy, tensorflow).
        If None (default), use random seed. Note that if you want completely deterministic
        results, you must set `n_cpu_tf_sess` to 1.
    :param n_cpu_tf_sess: (int) The number of threads for TensorFlow operations
        If None, the number of cpu of the current machine will be used.
    """

    def __init__(self, eval_env, policy, env, gamma=0.99, learning_rate=3e-4, buffer_size=50000,
                 learning_starts=100, train_freq=1, batch_size=64,
                 tau=0.005, ent_coef='auto', target_update_interval=1, episode_len=1000,
                 gradient_steps=1, target_entropy='auto', action_noise=None,
                 random_exploration=0.0, verbose=0, tensorboard_log=None,
                 _init_setup_model=True, policy_kwargs=None, full_tensorboard_log=False,
                 train_ent_sep=False, target_entropy_coef=1.0,
                 seed=None, n_cpu_tf_sess=None):

        super(SAC, self).__init__(policy=policy, env=env, replay_buffer=None, verbose=verbose,
                                  policy_base=SACPolicy, requires_vec_env=False, policy_kwargs=policy_kwargs,
                                  seed=seed, n_cpu_tf_sess=n_cpu_tf_sess)

        self.buffer_size = buffer_size
        self.episode_len = episode_len
        self.target_entropy_coef = target_entropy_coef
        self.learning_rate = learning_rate
        self.learning_starts = learning_starts
        self.train_freq = train_freq
        self.batch_size = batch_size
        self.eval_env = eval_env
        self.tau = tau
        # In the original paper, same learning rate is used for all networks.
        # self.policy_lr = learning_rate
        # self.qf_lr = learning_rate
        # self.vf_lr = learning_rate
        # Entropy coefficient / Entropy temperature
        # Inverse of the reward scale
        self.ent_coef = ent_coef
        self.target_update_interval = target_update_interval
        self.gradient_steps = gradient_steps
        self.gamma = gamma
        self.action_noise = action_noise
        self.random_exploration = random_exploration

        self.value_fn = None
        self.graph = None
        self.replay_buffer = None
        self.episode_reward = None
        self.sess = None
        self.tensorboard_log = tensorboard_log
        self.verbose = verbose
        self.params = None
        self.summary = None
        self.policy_tf = None
        self.target_entropy = target_entropy
        self.full_tensorboard_log = full_tensorboard_log

        self.train_ent_sep = train_ent_sep

        self.obs_target = None
        self.target_policy = None
        self.actions_ph = None
        self.rewards_ph = None
        self.terminals_ph = None
        self.observations_ph = None
        self.action_target = None
        self.next_observations_ph = None
        self.value_target = None
        self.step_ops = None
        self.target_update_op = None
        self.infos_names = None
        self.entropy = None
        self.target_params = None
        self.learning_rate_ph = None
        self.processed_obs_ph = None
        self.processed_next_obs_ph = None
        self.log_ent_coef = None

        if _init_setup_model:
            self.setup_model()

    def _get_pretrain_placeholders(self):
        policy = self.policy_tf
        # Rescale
        deterministic_action = self.deterministic_action * np.abs(self.action_space.low)
        return policy.obs_ph, self.actions_ph, deterministic_action

    def setup_model(self):
        summary_prefix = 'target/'
        with SetVerbosity(self.verbose):
            self.graph = tf.Graph()
            with self.graph.as_default():
                self.sess = tf_util.make_session(num_cpu=self.n_cpu_tf_sess, graph=self.graph)
                self.set_random_seed(self.seed)
                self.replay_buffer = ReplayBuffer(self.buffer_size)

                with tf.variable_scope("input", reuse=False):
                    # Create policy and target TF objects
                    self.policy_tf = self.policy(self.sess, self.observation_space, self.action_space,
                                                 **self.policy_kwargs)
                    self.target_policy = self.policy(self.sess, self.observation_space, self.action_space,
                                                     **self.policy_kwargs)

                    # Initialize Placeholders
                    self.observations_ph = self.policy_tf.obs_ph
                    # Normalized observation for pixels
                    self.processed_obs_ph = self.policy_tf.processed_obs
                    self.next_observations_ph = self.target_policy.obs_ph
                    self.processed_next_obs_ph = self.target_policy.processed_obs
                    self.action_target = self.target_policy.action_ph
                    self.terminals_ph = tf.placeholder(tf.float32, shape=(None, 1), name='terminals')
                    self.rewards_ph = tf.placeholder(tf.float32, shape=(None, 1), name='rewards')
                    self.target_ent_ph = tf.placeholder(tf.float32, shape=(), name='target_ent')
                    self.rewards_min_ph = tf.placeholder(tf.float32, shape=(), name='rewards')
                    self.rewards_max_ph = tf.placeholder(tf.float32, shape=(), name='rewards')
                    self.actions_ph = tf.placeholder(tf.float32, shape=(None,) + self.action_space.shape,
                                                     name='actions')
                    self.learning_rate_ph = tf.placeholder(tf.float32, [], name="learning_rate_ph")

                with tf.variable_scope("model", reuse=False):
                    # Create the policy
                    # first return value corresponds to deterministic actions
                    # policy_out corresponds to stochastic actions, used for training
                    # logp_pi is the log probabilty of actions taken by the policy
                    self.deterministic_action, policy_out, logp_pi = self.policy_tf.make_actor(self.processed_obs_ph)
                    # _, next_policy_out, next_logp_pi = self.policy_tf.make_actor(self.processed_obs_ph, add_action_ph=True, reuse=True)
                    # next_logp_pi = self.policy_tf.get_log_pi(self.next_observations_ph, self.)
                    # logp_pi = tf.reshape(logp_pi, (-1, 1))
                    # Monitor the entropy of the policy,
                    # this is not used for training
                    self.entropy = tf.reduce_mean(self.policy_tf.entropy)
                    #  Use two Q-functions to improve performance by reducing overestimation bias.
                    qf1, qf2, value_fn = self.policy_tf.make_critics(self.processed_obs_ph, self.actions_ph,
                                                                     create_qf=True, create_vf=True)
                    qf1_pi, qf2_pi, _ = self.policy_tf.make_critics(self.processed_obs_ph,
                                                                    policy_out, create_qf=True, create_vf=False, reuse=True)

                    # Target entropy is used when learning the entropy coefficient
                    if self.target_entropy == 'auto':
                        # automatically set target entropy if needed
                        self.target_entropy = self.target_ent_ph
                        # -np.prod(self.env.action_space.shape).astype(np.float32) * self.target_entropy_coef
                    else:
                        # Force conversion
                        # this will also throw an error for unexpected string
                        self.target_entropy = float(self.target_entropy)

                    # The entropy coefficient or entropy can be learned automatically
                    # see Automating Entropy Adjustment for Maximum Entropy RL section
                    # of https://arxiv.org/abs/1812.05905
                    if isinstance(self.ent_coef, str) and self.ent_coef.startswith('auto'):
                        # Default initial value of ent_coef when learned
                        init_value = 1.0
                        if '_' in self.ent_coef:
                            init_value = float(self.ent_coef.split('_')[1])
                            assert init_value > 0., "The initial value of ent_coef must be greater than 0"

                        self.log_ent_coef = tf.get_variable('log_ent_coef', dtype=tf.float32,
                                                            initializer=np.log(init_value).astype(np.float32))
                        self.ent_coef = tf.exp(self.log_ent_coef)
                    else:
                        # Force conversion to float
                        # this will throw an error if a malformed string (different from 'auto')
                        # is passed
                        self.ent_coef = float(self.ent_coef)

                with tf.variable_scope("target", reuse=False):
                    # Create the value network
                    _, _, value_target = self.target_policy.make_critics(self.processed_next_obs_ph,
                                                                         create_qf=False, create_vf=True)
                    self.value_target = value_target

                with tf.variable_scope("loss", reuse=False):
                    # Take the min of the two Q-Values (Double-Q Learning)
                    min_qf_pi = tf.minimum(qf1_pi, qf2_pi)
                    min_qf_pi = tf.clip_by_value(min_qf_pi, self.rewards_min_ph/(1-self.gamma), self.rewards_max_ph/(1-self.gamma))

                    # Target for Q value regression
                    q_backup = tf.stop_gradient(
                        self.rewards_ph +
                        (1 - self.terminals_ph) * self.gamma * self.value_target
                    )
                    q_backup = tf.clip_by_value(q_backup, self.rewards_min_ph/(1-self.gamma), self.rewards_max_ph/(1-self.gamma))
                    # Compute Q-Function loss
                    qf1_loss = 0.5 * tf.reduce_mean((q_backup - qf1) ** 2)
                    qf2_loss = 0.5 * tf.reduce_mean((q_backup - qf2) ** 2)

                    # Compute the entropy temperature loss
                    # it is used when the entropy coefficient is learned
                    ent_coef_loss, entropy_optimizer = None, None
                    if not isinstance(self.ent_coef, float):
                        ent_coef_loss = -tf.reduce_mean(
                            self.log_ent_coef * tf.stop_gradient(logp_pi + self.target_entropy))
                        entropy_optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate_ph)
                    # Compute the policy loss
                    # Alternative: policy_kl_loss = tf.reduce_mean(logp_pi - min_qf_pi)
                    policy_kl_loss = tf.reduce_mean(self.ent_coef * logp_pi - min_qf_pi)
                    # NOTE: in the original implementation, they have an additional
                    # regularization loss for the gaussian parameters
                    # this is not used for now
                    if hasattr(self.policy_tf, 'reg_loss') and self.policy_tf.reg_loss is not None:
                        policy_regularization_loss = self.policy_tf.reg_loss
                        policy_loss = (policy_kl_loss + policy_regularization_loss)
                    else:
                        logger.warn("[WARN] policy_regularization_loss not exist.")
                        policy_regularization_loss = None
                        policy_loss = policy_kl_loss
                    # policy_loss = policy_kl_loss


                    # Target for value fn regression
                    # We update the vf towards the min of two Q-functions in order to
                    # reduce overestimation bias from function approximation error.
                    self.ent_rew_op = self.ent_coef * logp_pi
                    v_backup = tf.stop_gradient(min_qf_pi - self.ent_rew_op)
                    value_loss = 0.5 * tf.reduce_mean((value_fn - v_backup) ** 2)
                    # qv_loss_total = (value_fn - v_backup) ** 2
                    values_losses = qf1_loss + qf2_loss + value_loss

                    # Policy train op
                    # (has to be separate from value train op, because min_qf_pi appears in policy_loss)
                    policy_optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate_ph)
                    policy_train_op = policy_optimizer.minimize(policy_loss, var_list=get_vars('model/pi'))

                    # Value train op
                    value_optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate_ph)
                    qf1_optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate_ph)
                    qf2_optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate_ph)
                    values_params = get_vars('model/values_fn')

                    source_params = get_vars("model/values_fn/vf")
                    target_params = get_vars("target/values_fn/vf")

                    # Polyak averaging for target variables
                    self.target_update_op = [
                        tf.assign(target, (1 - self.tau) * target + self.tau * source)
                        for target, source in zip(target_params, source_params)
                    ]
                    # Initializing target to match source variables
                    target_init_op = [
                        tf.assign(target, source)
                        for target, source in zip(target_params, source_params)
                    ]

                    # Control flow is used because sess.run otherwise evaluates in nondeterministic order
                    # and we first need to compute the policy action before computing q values losses
                    with tf.control_dependencies([policy_train_op]):
                        # train_values_op = value_optimizer.minimize(values_losses, var_list=values_params)
                        # use separate optimizers
                        train_values_op = value_optimizer.minimize(value_loss, var_list=get_vars('model/values_fn/vf'))
                        train_qf1_values_op = qf1_optimizer.minimize(qf1_loss, var_list=get_vars('model/values_fn/qf1'))
                        train_qf2_values_op = qf2_optimizer.minimize(qf2_loss, var_list=get_vars('model/values_fn/qf2'))

                        self.infos_names = ['policy_loss', 'min_qf_pi', 'rew_min', 'rew_max',
                                            'qf1_loss', 'qf2_loss', 'value_loss', 'entropy']
                        # All ops to call during one training step
                        self.step_ops = [policy_loss, min_qf_pi, self.rewards_min_ph, self.rewards_max_ph, qf1_loss, qf2_loss,
                                         value_loss, qf1, qf2, value_fn, logp_pi,
                                         self.entropy, policy_train_op, train_values_op, train_qf1_values_op, train_qf2_values_op]

                        self.explr_infos_names = ['policy_loss', 'qf1_loss', 'qf2_loss','entropy']
                        # All ops to call during one training step
                        self.explr_step_ops = [policy_loss, qf1_loss, qf2_loss,
                                         qf1, qf2, logp_pi,
                                         self.entropy, policy_train_op, train_qf1_values_op, train_qf2_values_op]

                        # Add entropy coefficient optimization operation if needed
                        if ent_coef_loss is not None:
                            with tf.control_dependencies([train_values_op, train_qf1_values_op, train_qf2_values_op]):
                                ent_coef_op = entropy_optimizer.minimize(ent_coef_loss, var_list=self.log_ent_coef)
                                if self.train_ent_sep:
                                    self.ent_step_ops = [ent_coef_op, ent_coef_loss, self.ent_coef]
                                    self.ent_infos_names = ['ent_coef_loss', 'ent_coef']
                                else:
                                    self.infos_names += ['ent_coef_loss', 'ent_coef']
                                    self.explr_infos_names += ['ent_coef_loss', 'ent_coef']
                                    self.step_ops += [ent_coef_op, ent_coef_loss, self.ent_coef]
                                    self.explr_step_ops += [ent_coef_op, ent_coef_loss, self.ent_coef]
                    self.summary_ops = [policy_loss, policy_kl_loss,
                                        qf1_loss, qf2_loss, value_loss, self.entropy, tf.reduce_mean(self.learning_rate_ph)]
                    # self.additional_ops = [qv_loss_total, value_fn, v_backup, min_qf_pi, logp_pi, self.ent_coef, self.ent_coef * logp_pi]
                    self.summary_name = ["policy_loss",
                                         "policy_kl_loss",
                                         "qf1_loss", "qf2_loss", "value_loss", "entropy", "learning_rate"]
                    if policy_regularization_loss is not None:
                        self.summary_ops.append(policy_regularization_loss)
                        self.summary_name.append("policy_regularization_loss")
                        # Monitor losses and entropy in tensorboard
                    # tf.summary.scalar(summary_prefix + 'policy_loss', policy_loss)
                    # tf.summary.scalar(summary_prefix + 'qf1_loss', qf1_loss)
                    # tf.summary.scalar(summary_prefix + 'qf2_loss', qf2_loss)
                    # tf.summary.scalar(summary_prefix + 'value_loss', value_loss)
                    # tf.summary.scalar(summary_prefix + 'entropy', self.entropy)
                    if ent_coef_loss is not None:
                        real_entropy = - tf.reduce_mean(logp_pi)
                        self.summary_ops += [ent_coef_loss, self.ent_coef, real_entropy]
                        self.summary_name += ['ent_coef_loss', 'ent_coef', 'pilogpi']
                        # tf.summary.scalar(summary_prefix + 'ent_coef_loss', ent_coef_loss)
                        # tf.summary.scalar(summary_prefix + 'ent_coef', self.ent_coef)


                # Retrieve parameters that must be saved
                self.params = get_vars("model")
                self.target_params = get_vars("target/values_fn/vf")

                # Initialize Variables and target network
                with self.sess.as_default():
                    self.sess.run(tf.global_variables_initializer())
                    self.sess.run(target_init_op)

    def ent_rew(self, obs):
        return self.sess.run(self.ent_rew_op, feed_dict={self.processed_obs_ph: obs})

    def _train_step(self, batch, step, learning_rate, current_target_ent, batch_size, rewards_min, rewards_max,
                    prefix_name='train_step/', explr_train=False):
        # Sample a batch from the replay buffer
        batch_obs, batch_actions, batch_rewards, batch_next_obs, batch_dones = batch

        feed_dict = {
            self.observations_ph: batch_obs,
            self.actions_ph: batch_actions,
            self.next_observations_ph: batch_next_obs,
            self.rewards_ph: batch_rewards.reshape(batch_size, -1),
            self.terminals_ph: batch_dones.reshape(batch_size, -1),
            self.learning_rate_ph: learning_rate,
            self.target_ent_ph: current_target_ent,
            self.rewards_max_ph: rewards_max,
            self.rewards_min_ph: rewards_min
        }

        # out  = [policy_loss, qf1_loss, qf2_loss,
        #         value_loss, qf1, qf2, value_fn, logp_pi,
        #         self.entropy, policy_train_op, train_values_op]

        # Do one gradient step
        # and optionally compute log for tensorboard

        out = self.sess.run([self.summary_ops] + self.step_ops, feed_dict)
        summary = out.pop(0)
        policy_loss, qf1_loss, qf2_loss, value_loss, *values = out
        if step % (self.train_freq * 2 * 500) == 0:
            # logger.log_from_tf_summary(summary)
            logger.log_key_value(self.summary_name + self.infos_names, summary, prefix_name=prefix_name)
            logger.record_tabular(prefix_name + "policy_loss", np.mean(policy_loss))
            logger.dump_tabular()
        # Unpack to monitor losses and entropy
        # qf1, qf2, value_fn, logp_pi, entropy, *_ = values
        entropy = values[4]

        if self.log_ent_coef is not None:
            ent_coef_loss, ent_coef = values[-2:]
            return policy_loss, qf1_loss, qf2_loss, value_loss, entropy, ent_coef_loss, ent_coef
        return policy_loss, qf1_loss, qf2_loss, value_loss, entropy, summary

    def _ent_train_step(self, batch, step, learning_rate, batch_size, prefix_name=''):
        # Sample a batch from the replay buffer
        if self.train_ent_sep:
            batch_obs, batch_actions, batch_rewards, batch_next_obs, batch_dones = batch

            feed_dict = {   
                self.observations_ph: batch_obs,
                self.actions_ph: batch_actions,
                self.next_observations_ph: batch_next_obs,
                self.rewards_ph: batch_rewards.reshape(batch_size, -1),
                self.terminals_ph: batch_dones.reshape(batch_size, -1),
                self.learning_rate_ph: learning_rate
            }
            out = self.sess.run(self.ent_step_ops, feed_dict)
    #
    # def _get_qv_loss_total(self, batch):
    #     # Sample a batch from the replay buffer
    #     batch_obs, batch_actions, batch_rewards, batch_next_obs, batch_dones = batch
    #     #self.policy_tf.step(obs[None], deterministic=False).flatten()
    #     batch_actions = self.policy_tf.step(batch_obs, deterministic=False)
    #     feed_dict = {
    #         self.observations_ph: batch_obs,
    #         self.actions_ph: batch_actions,
    #         self.next_observations_ph: batch_next_obs,
    #         self.rewards_ph: batch_rewards.reshape(-1, 1),
    #         self.terminals_ph: batch_dones.reshape(-1, 1),
    #     }
    #
    #     # out  = [policy_loss, qf1_loss, qf2_loss,
    #     #         value_loss, qf1, qf2, value_fn, logp_pi,
    #     #         self.entropy, policy_train_op, train_values_op]
    #
    #     # Do one gradient step
    #     # and optionally compute log for tensorboard
    #     out = self.sess.run([self.additional_ops], feed_dict)
    #     qv_loss_total = out[0][0]
    #     #value_fn = out[0][1]
    #     #v_back = out[0][2]
    #     #min_q = out[0][3]
    #     #log_pi = out[0][4]
    #     #ent_coef = out[0][5]
    #     #ent_coef_logp_pi = out[0][6]
    #
    #     return qv_loss_total

    def learn(self, total_timesteps, rew_scale=1, callback=None, log_interval=4, dm_env=None,
              tb_log_name="SAC", reset_num_timesteps=True, replay_wrapper=None):

        new_tb_log = self._init_num_timesteps(reset_num_timesteps)
        if replay_wrapper is not None:
            self.replay_buffer = replay_wrapper(self.replay_buffer)

        with SetVerbosity(self.verbose):

            self._setup_learn()

            # Transform to callable if needed
            self.learning_rate = get_schedule_fn(self.learning_rate)
            # Initial learning rate
            current_lr = self.learning_rate(1)

            start_time = time.time()
            episode_rewards = [0.0]
            episode_successes = []
            if self.action_noise is not None:
                self.action_noise.reset()
            if dm_env is not None:
                obs = dm_env.reset()
            else:
                obs = self.env.reset()
            self.episode_reward = np.zeros((1,))
            ep_info_buf = deque(maxlen=100)
            n_updates = 0
            infos_values = []
            epi_steps = 0
            rew_min = np.inf
            rew_max = 1 # should be larger than 0
            for step in range(total_timesteps):
                if callback is not None:
                    # Only stop training if return value is False, not when it is None. This is for backwards
                    # compatibility with callbacks that have no return statement.
                    if callback(locals(), globals()) is False:
                        break
                exp_manager.time_step_holder.set_time(step)
                # Before training starts, randomly sample actions
                # from a uniform distribution for better exploration.
                # Afterwards, use the learned policy
                # if random_exploration is set to 0 (normal setting)
                if (self.num_timesteps < self.learning_starts
                    or np.random.rand() < self.random_exploration):
                    # No need to rescale when sampling random action
                    rescaled_action = action = self.env.action_space.sample()
                else:
                    # self.predict(obs[None])
                    action = self.policy_tf.step(obs[None], deterministic=False).flatten()
                    # Add noise to the action (improve exploration,
                    # not needed in general)

                    # Rescale from [-1, 1] to the correct bounds
                    if self.action_noise is not None:
                        rescaled_action = np.clip(action + self.action_noise(), -1, 1)
                        rescaled_action = rescaled_action * np.abs(self.action_space.low)
                    else:
                        rescaled_action = action * np.abs(self.action_space.low)
                assert action.shape == self.env.action_space.shape
                if dm_env is not None:
                    new_obs, reward, done, info = dm_env.step(rescaled_action)
                else:
                    new_obs, reward, done, info = self.env.step(rescaled_action)
                # done = False
                # assert done == False
                epi_steps += 1
                reward *= rew_scale
                rew_min = np.minimum(reward, rew_min)
                rew_max = np.maximum(reward, rew_max)
                # Store transition in the replay buffer.
                self.replay_buffer.add(obs, action, reward, new_obs, float(done))
                obs = new_obs

                # Retrieve reward and episode length if using Monitor wrapper
                maybe_ep_info = info.get('episode')
                if maybe_ep_info is not None:
                    ep_info_buf.extend([maybe_ep_info])
                # 这里强行转化成向量形式，是为了适应原本的多线程的调用
                ep_reward = np.array([reward]).reshape((1, -1))
                exp_manager.time_step_holder.set_time(step)

                if step % self.train_freq == 0:
                    mb_infos_vals = []
                    # Update policy, critics and target networks
                    for grad_step in range(self.gradient_steps):
                        # Break if the warmup phase is not over
                        # or if there are not enough samples in the replay buffer
                        if not self.replay_buffer.can_sample(self.episode_len) \
                           or self.num_timesteps < self.learning_starts:
                            break
                        n_updates += 1
                        # Compute current learning_rate
                        frac = 1.0 - step / total_timesteps
                        current_lr = self.learning_rate(frac)
                        current_target_ent = -np.prod(self.env.action_space.shape).astype(np.float32) * (self.target_entropy_coef)
                        # Update policy and critics (q functions)

                        batch = self.replay_buffer.sample(self.batch_size)
                        mb_infos_vals.append(self._train_step(batch, step, current_lr, current_target_ent, self.batch_size,
                                                              rewards_min=rew_min, rewards_max=rew_max))
                        # Update target network
                        if (step + grad_step) % self.target_update_interval == 0:
                            # Update target network
                            self.sess.run(self.target_update_op)
                    # Log losses and entropy, useful for monitor training
                    # if len(mb_infos_vals) > 0:
                    #     infos_values = np.mean(mb_infos_vals, axis=0)

                episode_rewards[-1] += reward
                if done or epi_steps >= self.episode_len:
                    finish_epi = True
                    logger.record_tabular("episode/len", epi_steps)
                    epi_steps = 0
                    if self.action_noise is not None:
                        self.action_noise.reset()
                    # if not isinstance(self.env, VecEnv):
                    if dm_env is not None:
                        obs = dm_env.reset()
                    else:
                        obs = self.env.reset()
                    episode_rewards.append(0.0)

                    maybe_is_success = info.get('is_success')
                    if maybe_is_success is not None:
                        episode_successes.append(float(maybe_is_success))
                else:
                    finish_epi = False

                ep_done = np.array([finish_epi]).reshape((1, -1))
                self.episode_reward = calculate_episode_reward_logger(self.episode_reward, ep_reward,
                                                                  ep_done, self.num_timesteps)
                if len(episode_rewards[-101:-1]) == 0:
                    mean_reward = -np.inf
                else:
                    mean_reward = round(float(np.mean(episode_rewards[-101:-1])), 1)

                num_episodes = len(episode_rewards)
                # Display training infos
                if self.num_timesteps % 4000 == 0:
                    if dm_env is not None:
                        eval_ob = dm_env.reset()
                        all_state = np.insert(eval_ob, 0, 0)
                        qpos = all_state[:int(all_state.shape[0] / 2)]
                        qvel = all_state[int(all_state.shape[0] / 2):]
                        self.eval_env.envs[0].reset()
                        self.eval_env.envs[0].set_state(qpos, qvel)
                    else:
                        eval_ob = self.eval_env.reset()
                    eval_epi_rewards = 0
                    eval_epis = 0
                    eval_performance = []
                    eval_epi_steps = 0
                    while True:
                        eval_rescaled_action, _ = self.predict(eval_ob[None])
                        # eval_action = self.policy_tf.step(eval_ob[None], deterministic=True).flatten()
                        # eval_rescaled_action = eval_action * np.abs(self.action_space.low)
                        if self.num_timesteps % 120000 == 0 and eval_epis < 2:
                            img = self.eval_env.render(mode='rgb_array')
                            ImgRecorder.save(name="imgs/{}-{}-{}.jpg".format(self.num_timesteps, eval_epis, eval_epi_steps), img=img)
                        # exp_manager.results_dir
                        eval_new_obs, eval_reward, eval_done, eval_info = self.eval_env.step(eval_rescaled_action)
                        if dm_env is not None:
                            all_state = np.insert(eval_new_obs, 0, 0)
                            qpos = all_state[:int(all_state.shape[0] / 2)]
                            qvel = all_state[int(all_state.shape[0] / 2):]
                            self.eval_env.envs[0].reset()
                            self.eval_env.envs[0].set_state(qpos, qvel)
                        eval_epi_rewards += eval_reward
                        eval_ob = eval_new_obs
                        eval_epi_steps += 1
                        if eval_done or eval_epi_steps >= self.episode_len:
                            if dm_env is not None:
                                eval_ob = dm_env.reset()
                                all_state = np.insert(eval_ob, 0, 0)
                                qpos = all_state[:int(all_state.shape[0] / 2)]
                                qvel = all_state[int(all_state.shape[0] / 2):]
                                self.eval_env.envs[0].reset()
                                self.eval_env.envs[0].set_state(qpos, qvel)
                            else:
                                eval_ob = self.eval_env.reset()
                            eval_performance.append(eval_epi_rewards)
                            eval_epi_rewards = 0
                            eval_epis += 1
                            eval_epi_steps = 0
                            if eval_epis > 50:
                                break
                    logger.record_tabular("eval/performance", np.mean(eval_performance))
                    if dm_env is not None and self.num_timesteps % 40000 == 0:
                        eval_ob = dm_env.reset()
                        eval_epi_rewards = 0
                        eval_epis = 0
                        eval_performance = []
                        eval_epi_steps = 0
                        while True:
                            eval_rescaled_action, _ = self.predict(eval_ob[None])
                            # eval_action = self.policy_tf.step(eval_ob[None], deterministic=True).flatten()
                            # eval_rescaled_action = eval_action * np.abs(self.action_space.low)
                            if self.num_timesteps % 120000 == 0 and eval_epis < 2:
                                img = dm_env.render(mode='rgb_array')
                                ImgRecorder.save(name="imgs-dm/{}-{}-{}.jpg".format(self.num_timesteps, eval_epis, eval_epi_steps), img=img)
                            # exp_manager.results_dir
                            eval_new_obs, eval_reward, eval_done, eval_info = dm_env.step(eval_rescaled_action)
                            eval_epi_rewards += eval_reward
                            eval_ob = eval_new_obs
                            eval_epi_steps += 1
                            if eval_done or eval_epi_steps >= self.episode_len:
                                eval_ob = dm_env.reset()
                                eval_performance.append(eval_epi_rewards)
                                eval_epi_rewards = 0
                                eval_epis += 1
                                eval_epi_steps = 0
                                if eval_epis > 50:
                                    break
                        logger.record_tabular("eval/performance_dm", np.mean(eval_performance))

                if self.verbose >= 1 and finish_epi and log_interval is not None and len(episode_rewards) % log_interval == 0:
                    fps = int(step / (time.time() - start_time))
                    logger.logkv("episodes", num_episodes)
                    logger.logkv("mean 100 episode reward", mean_reward)
                    if len(ep_info_buf) > 0 and len(ep_info_buf[0]) > 0:
                        logger.logkv('ep_rewmean', safe_mean([ep_info['r'] for ep_info in ep_info_buf]))
                        logger.logkv('eplenmean', safe_mean([ep_info['l'] for ep_info in ep_info_buf]))
                    logger.logkv("n_updates", n_updates)
                    logger.logkv("current_lr", current_lr)
                    logger.logkv("fps", fps)
                    logger.logkv('time_elapsed', int(time.time() - start_time))
                    if len(episode_successes) > 0:
                        logger.logkv("success rate", np.mean(episode_successes[-100:]))
                    # if len(infos_values) > 0:
                    #     for (name, val) in zip(self.infos_names, infos_values):
                    #         logger.logkv(name, val)
                    logger.logkv("total timesteps", self.num_timesteps)
                    logger.dumpkvs()
                    # Reset infos:
                    infos_values = []
                self.num_timesteps += 1
            return self

    def action_probability(self, observation, state=None, mask=None, actions=None, logp=False):
        if actions is not None:
            raise ValueError("Error: SAC does not have action probabilities.")

        warnings.warn("Even though SAC has a Gaussian policy, it cannot return a distribution as it "
                      "is squashed by a tanh before being scaled and ouputed.")

        return None

    def predict(self, observation, state=None, mask=None, deterministic=True):
        observation = np.array(observation)
        vectorized_env = self._is_vectorized_observation(observation, self.observation_space)

        observation = observation.reshape((-1,) + self.observation_space.shape)
        actions, std = self.policy_tf.step(observation, deterministic=deterministic)
        actions = actions.reshape((-1,) + self.action_space.shape)  # reshape to the correct action shape
        actions = actions * np.abs(self.action_space.low)  # scale the output for the prediction

        if not vectorized_env:
            actions = actions[0]

        return actions, std

    def get_parameter_list(self):
        return (self.params +
                self.target_params)

    def save(self, save_path, cloudpickle=False):
        data = {
            "learning_rate": self.learning_rate,
            "buffer_size": self.buffer_size,
            "learning_starts": self.learning_starts,
            "train_freq": self.train_freq,
            "batch_size": self.batch_size,
            "tau": self.tau,
            "ent_coef": self.ent_coef if isinstance(self.ent_coef, float) else 'auto',
            "target_entropy": self.target_entropy,
            # Should we also store the replay buffer?
            # this may lead to high memory usage
            # with all transition inside
            # "replay_buffer": self.replay_buffer
            "gamma": self.gamma,
            "verbose": self.verbose,
            "observation_space": self.observation_space,
            "action_space": self.action_space,
            "policy": self.policy,
            "n_envs": self.n_envs,
            "n_cpu_tf_sess": self.n_cpu_tf_sess,
            "seed": self.seed,
            "action_noise": self.action_noise,
            "random_exploration": self.random_exploration,
            "_vectorize_action": self._vectorize_action,
            "policy_kwargs": self.policy_kwargs
        }

        params_to_save = self.get_parameters()

        self._save_to_file(save_path, data=data, params=params_to_save, cloudpickle=cloudpickle)
