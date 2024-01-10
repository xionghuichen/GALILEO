from GALILEO.offline_data.dataset_handler import DatasetHandler
import tensorflow as tf
class BaseLearner:
    def __init__(self, dataset_holder):
        self.dataset_holder = dataset_holder
        assert isinstance(self.dataset_holder, DatasetHandler)
        self.s_dim = dataset_holder.s_dim
        self.n_actions = dataset_holder.a_dim
        self.ps_dim = dataset_holder.ps_dim
        self.init_place_holder()
    
    def init_place_holder(self):
        self.s_input_ph = tf.placeholder(dtype=tf.float32, shape=[None, self.s_dim], name='s')
        self.a_input_ph = tf.placeholder(dtype=tf.float32, shape=[None, self.n_actions], name='a')
        self.p_s_next_target_ph = tf.placeholder(dtype=tf.float32, shape=[None, self.ps_dim], name='ps')
        
    
    def norm_s(self, f_s):
        return (f_s - self.dataset_holder.s_mean) / self.dataset_holder.s_std

    def norm_a(self, f_a):
        return (f_a - self.dataset_holder.a_mean) / self.dataset_holder.a_std

    def norm_p_s_next(self, f_p_s_next):
        return (f_p_s_next - self.dataset_holder.p_s_n_mean) / self.dataset_holder.p_s_n_std
    
    def denorm_p_s_next(self, norm_p_s_next):
        return norm_p_s_next * self.dataset_holder.p_s_n_std + self.dataset_holder.p_s_n_mean
    
    def denorm_s(self, norm_s):
        return norm_s *  self.dataset_holder.s_std + self.dataset_holder.s_mean
    
    def denorm_a(self, norm_a):
        return norm_a *  self.dataset_holder.a_std + self.dataset_holder.a_mean