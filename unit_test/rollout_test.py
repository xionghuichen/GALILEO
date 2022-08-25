# Created by xionghuichen at 2022/7/26
# Email: chenxh@lamda.nju.edu.cn


from GALILEO.models.nets import *
seed = 1
rng = jax.random.PRNGKey(seed)


def pi_func(x):
    module = PiModule([12, 12], 1)
    return module(x)

def M_func(x, a):
    module = DynamicsModule([12, 12], 2)
    return module(x, a)

s = np.array([[1.,1.]])
a = np.array([[1.]])
pi = hk.without_apply_rng(hk.transform(pi_func))
M = hk.without_apply_rng(hk.transform(M_func))
pi_param = pi.init(rng=1, x=s)
M_param = M.init(rng=1, x=s, a=a)
pi_frwd = jax.jit(pi.apply)
M_frwd = jax.jit(M.apply)


@jax.jit
def sample(param, s, rng):
    mu, sig = pi_frwd(param, s)
    dist = distrax.MultivariateNormalDiag(mu, sig)
    return dist.sample(seed=rng)

class Policy:
    def __init__(self, param):
        self.param = param


    def sample(self, s, rng):
        return sample(self.param, s, rng)

policy = Policy(pi_param)

@jax.jit
def rollout(s_start, pi_param, M_param, rng):
    x = s_start
    for i in range(10):
        rng, subrng = jax.random.split(rng, 2)
        mu, sig, dist = pi_frwd(pi_param, x)
        a = dist.sample(seed=subrng)
        m_mu, m_sig, m_dist = M_frwd(M_param, x, a)
        rng, subrng = jax.random.split(rng, 2)
        x = m_dist.sample(seed=subrng)