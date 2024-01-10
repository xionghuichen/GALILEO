# Created by xionghuichen at 2022/9/5
# Email: chenxh@lamda.nju.edu.cn
import gym


class MjEnv(object):
    def __init__(self, env, res_pred=False):
        self.env = env
        self.res_pred = res_pred
        pass

    def complete_env(self, s, a, ps):
        ps = ps[..., 1:]  # remove rewards
        if self.res_pred:
            return s + ps
        else:
            return ps
