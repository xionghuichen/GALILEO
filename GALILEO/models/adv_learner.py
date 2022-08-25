# Created by xionghuichen at 2022/7/25
# Email: chenxh@lamda.nju.edu.cn

import haiku as hk
import numpy as np
from GALILEO.models.nets import *

class GalileoModule:
    def __init__(self, pi, V, M, D1, D2, map_func):
        self.pi = pi
        self.V = V
        self.M = M
        self.D1 = D1
        self.D2 = D2
        self.map_func = map_func
        pass

    def rollout(self, horizon, s_start):
        s = s_start
        res = []
        for h in range(horizon):
            a = self.pi(s)
            y = self.M(np.concatenate([s,a], axis=-1))
            s_next = self.map_func(s, a, y)
            res.append
