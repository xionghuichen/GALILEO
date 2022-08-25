# Created by xionghuichen at 2022/7/25
# Email: chenxh@lamda.nju.edu.cn
import numpy as np
from GALILEO.config import *

class GnfcEnv(object):
    def __init__(self, dim, dm_noise, one_step_dyn, target_line):
        self.dim = dim
        self.dm_noise = dm_noise
        self.one_step_dyn = one_step_dyn
        self.target_line = target_line

    def reset(self, batch_size):
        self.batch_size = batch_size
        s = np.array([np.random.normal(np.ones(self.dim - 1) * i / batch_size * MAX_S_MEAN, np.ones(self.dim - 1))
                      for i in range(batch_size)])
        s = np.append(s, np.zeros([s.shape[0], 1]), axis=-1)
        return s

    def part_env(self, s, a):
        mean = np.mean(s[..., :-1], axis=-1, keepdims=True) + 1.0 * a
        noise = np.random.normal(0, self.dm_noise, mean.shape)
        return mean + noise

    def complete_env(self, s, a, ps_next):
        if self.one_step_dyn:
            return s
        else:
            s_new = s[..., :-1] - np.mean(s[..., :-1], axis=-1, keepdims=True) + ps_next
            s_hist = s[..., -1:].copy()
            s_hist += ps_next
        return np.append(s_new, s_hist, axis=-1)

    def step(self, s, a):
        ps = self.part_env(s, a)
        next_s = self.complete_env(s, a, ps)
        r = np.square((ps-self.target_line))
        return next_s, r, np.zeros(r.shape), ps


class GnfcPolicy:
    def __init__(self, target_line, random_prob, noise_scale):
        self.target_line = target_line
        self.random_prob = random_prob
        self.noise_scale = noise_scale

    def act(self, s):
        mean = (self.target_line - np.mean(s[..., :-1], axis=-1, keepdims=True)) / 15.0
        if np.random.random() < self.random_prob:
            noise = np.random.uniform(- self.noise_scale, self.noise_scale, mean.shape)
        else:
            noise = 0.
        return mean + noise
