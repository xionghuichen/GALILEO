# Created by xionghuichen at 2022/10/27
# Email: chenxh@lamda.nju.edu.cn

import gym
import numpy as np
from GALILEO.learner.base import BaseLearner


class DMEnv(BaseLearner):
    def __init__(self, dm_model, data_handler, sess, terminal_fn, gym_env, branch_init, episode_len,
                 deter_pred=True, use_real_env=False, real_reset=False, state_cons=True, acs_cons_scale=2.0):
        self.dm_model = dm_model
        self.sess = sess
        self.terminal_fn = terminal_fn
        self.gym_env = gym_env
        self.mean_pred_error = 0
        self.state_cons = state_cons
        self.branch_init = branch_init
        self.deter_pred = deter_pred
        self.episode_len = episode_len
        self.real_reset = real_reset
        self.acs_cons_scale = acs_cons_scale
        self.use_real_env = use_real_env
        super(DMEnv, self).__init__(data_handler)
        self.s_max = self.dataset_holder.traj_S.max(axis=(0,1))
        self.s_min = self.dataset_holder.traj_S.min(axis=(0,1))
        # self.s_max = self.dataset_holder.dataset.smax
        # self.s_min = self.dataset_holder.dataset.smin

        pass

    def graph_construction(self):
        norm_gen_s_input = self.norm_s(self.s_input_ph)
        norm_gen_a_input = self.norm_a(self.a_input_ph)
        self.dm_dist, self.dm_mean, self.dm_std = self.dm_model.obj_graph_construct(
            (norm_gen_s_input, norm_gen_a_input))
        self.dm_sample = self.dm_dist.sample()

    def evaluate_next_state(self, s, a, deter=True):
        norm_ps_next = self.sess.run(self.dm_mean, feed_dict={self.s_input_ph: s, self.a_input_ph: a})
        denorm_ps_next = self.denorm_p_s_next(norm_ps_next)
        if not deter:
            denorm_ps_next += np.random.normal(0, 0.1, denorm_ps_next.shape)
        return self.denorm_p_s_next(norm_ps_next)

    def reset(self, batch=0):
        if self.real_reset:
            obs = self.gym_env.reset()
            self.s = np.array([obs])
        elif self.branch_init:
            S_start = self.dataset_holder.traj_S[:-self.episode_len].reshape([-1, self.dataset_holder.s_dim])
            self.sample_idx = np.random.randint(0, S_start.shape[0], 1)
            self.s = S_start[self.sample_idx]
        else:
            S_start = self.dataset_holder.traj_S[0]
            self.sample_idx = np.random.randint(0, S_start.shape[0], 1)
            self.s = S_start[self.sample_idx]
        return self.s[0]

    def step(self, a):
        single_sample_mode = False
        if len(a.shape) == 1 or ():
            single_sample_mode = True
            a = np.expand_dims(a, axis=0)
        elif len(a.shape) == 2 and a.shape[0] == 1:
            single_sample_mode = True

        all_state = np.insert(self.s[0], 0, 0)
        qpos = all_state[:int(all_state.shape[0]/2)]
        qvel = all_state[int(all_state.shape[0]/2):]
        self.gym_env.reset()
        self.gym_env.set_state(qpos, qvel)
        real_ob, real_rew, real_done, info = self.gym_env.step(a[0])
        if not self.use_real_env:
            denorm_ps_next = self.evaluate_next_state(self.s, a, deter=self.deter_pred)
            rew = denorm_ps_next[..., 0:1]
            next_s = denorm_ps_next[..., 1:]
            done = self.terminal_fn(self.s, a, next_s)
        else:
            next_s = [real_ob]
            done = [[real_done],]
        assert single_sample_mode

        self.s = next_s
        # 0.08 is roughly the std of an imitated policy,
        # modeled with gaussian distribution and trained in the medium dataset.
        if self.acs_cons_scale > 0 and np.abs(self.dataset_holder.traj_A - a).max(-1).min() > 0.08 * self.acs_cons_scale:
            done = True
            rew = -5
        elif self.state_cons and (np.any(next_s - self.s_max > 0) or np.any(next_s - self.s_min < 0)):
            done = True
            rew = -5
        else:
            done = done[0][0]
            # rew = real_rew
            rew =  rew[0][0]
        # if (np.any(next_s - self.s_max * 1.3 > 0) or np.any(next_s - self.s_min * 1.3 < 0)):
        #     ood = True
        #     ood_info = f'a [{self.dataset_holder.traj_A.min(axis=(0, 1))}, {a} , {self.dataset_holder.traj_A.max(axis=(0, 1))}], \ns: [{self.s_min}, {next_s}, {self.s_max}'
        # else:
        #     ood = False
        #     ood_info = ''
        #
        # if np.abs(self.dataset_holder.traj_A - a).max(-1).min() > 0.16 or (np.any(next_s - self.s_max > 0) or np.any(next_s - self.s_min < 0)):
        #     ood = True
        # else:
        #     ood = False
        # if np.mean(np.square(self.dataset_holder.traj_S - next_s), axis=-1).min() > self.mean_pred_error:
        #     done = True
        # else:
        #     done = done[0, 0]
        # if single_sample_mode:
        next_s = next_s[0]

        return next_s, rew, done, {} # {'ood': ood, 'ood_info': ood_info}

    def render(self, mode):
        all_state = np.insert(self.s[0], 0, 0)
        qpos = all_state[:int(all_state.shape[0]/2)]
        qvel = all_state[int(all_state.shape[0]/2):]
        self.gym_env.reset()
        self.gym_env.set_state(qpos, qvel)
        img = self.gym_env.render(mode='rgb_array')
        return img

