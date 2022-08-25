# Created by xionghuichen at 2022/7/26
# Email: chenxh@lamda.nju.edu.cn

from GALILEO.offline_data.dataloader import Dataset
from GALILEO.config import *
from GALILEO.utils import *
from GALILEO.envs.gnfc import GnfcEnv
from RLA import logger
import numpy as np


class DatasetHandler(object):
    def __init__(self, dataset):
        assert isinstance(dataset, Dataset)
        self.traj_S = dataset.traj_S
        self.traj_A = dataset.traj_A
        self.traj_P_S_next = dataset.traj_P_S_next
        self.s_mean = dataset.s_min
        self.s_std = dataset.s_max - dataset.s_min
        self.s_std[self.s_std < 1.0] = 1.0
        self.a_mean = dataset.a_min
        self.a_std = (dataset.a_max - dataset.a_min) * 1.2
        self.p_s_n_mean = dataset.psmin
        self.p_s_n_std = (dataset.psmax - dataset.psmin) * 1.2