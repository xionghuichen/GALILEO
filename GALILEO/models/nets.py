# Created by xionghuichen at 2022/7/26
# Email: chenxh@lamda.nju.edu.cn
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import distrax

LOG_STD_MAX = 2
LOG_STD_MIN = -10


class DisModule(hk.Module):
    def __init__(self, hidden_states, name=None):
        super(DisModule, self).__init__(name)
        self.hidden_states = hidden_states

    def __call__(self, x):
        skeleton = []
        for h in self.hidden_states:
            skeleton.extend([hk.Linear(h), jax.nn.leaky_relu])
        skeleton.append(hk.Linear(1))
        return hk.Sequential(skeleton)(x)

class ValueModule(hk.Module):
    def __init__(self, hidden_states, name=None):
        super(ValueModule, self).__init__(name)
        self.hidden_states = hidden_states

    def __call__(self, x):
        skeleton = []
        for h in self.hidden_states:
            skeleton.extend([hk.Linear(h), jax.nn.leaky_relu])
        skeleton.append(hk.Linear(1))
        return hk.Sequential(skeleton)(x)


class PiModule(hk.Module):
    def __init__(self, hidden_states, action_n, rescale=1.2):
        super(PiModule, self).__init__()
        self.rescale = rescale
        self.hidden_states = hidden_states
        self.action_n = action_n

    def __call__(self, feat):
        skeleton = []
        for h in self.hidden_states:
            skeleton.extend([hk.Linear(h), jax.nn.leaky_relu])
        skeleton.extend([hk.Linear(self.action_n), jax.nn.tanh])
        mu = hk.Sequential(skeleton)(feat) * self.rescale
        log_std = hk.get_parameter("log_std", shape=[self.action_n], init=lambda shape, dtype: np.ones(shape, dtype) * -1.3)
        log_std = jnp.clip(log_std, LOG_STD_MIN, LOG_STD_MAX)
        sig = jnp.exp(log_std)
        return mu, sig
        # , distrax.MultivariateNormalDiag(mu, sig)


class DynamicsModule(hk.Module):
    def __init__(self, hidden_states, dim_next_x, rescale=1.2, name=None):
        super(DynamicsModule, self).__init__(name)
        self.rescale = rescale
        self.hidden_states = hidden_states
        self.dim_next_x = dim_next_x

    def __call__(self, x, a):
        skeleton = []
        xa = np.concatenate([x, a], axis=-1)
        for h in self.hidden_states:
            skeleton.extend([hk.Linear(h), jax.nn.leaky_relu])
        skeleton.extend([hk.Linear(self.dim_next_x), jax.nn.tanh])
        mu = hk.Sequential(skeleton)(xa) * self.rescale
        log_std = hk.get_parameter("log_std", shape=[self.dim_next_x], init=lambda shape, dtype: np.ones(shape, dtype)*-1.3)
        log_std = jnp.clip(log_std, LOG_STD_MIN, LOG_STD_MAX)
        sig = jnp.exp(log_std)
        return mu, sig
        # , distrax.MultivariateNormalDiag(mu, sig)
