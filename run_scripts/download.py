# Created by xionghuichen at 2023/4/28
# Email: chenxh@lamda.nju.edu.cn
import tensorflow as tf
import os
import pickle

if __name__ == '__main__':
    with tf.io.gfile.GFile(os.path.join('gs://gresearch/deep-ope/d4rl/hopper/hopper_ood_0.pkl'), 'rb') as f:
        weights = pickle.load(f)