# Created by xionghuichen at 2022/7/26
# Email: chenxh@lamda.nju.edu.cn
import chex
import jax
import jax.numpy as jnp
Array = chex.Array
Scalar = chex.Scalar

#
# def policy_gradient_loss(
#     logits_t: Array,
#     a_t: Array,
#     adv_t: Array,
#     w_t: Array,
#     use_stop_gradient: bool = True,
# ) -> Array:
#   """Calculates the policy gradient loss.
#   See "Simple Gradient-Following Algorithms for Connectionist RL" by Williams.
#   (http://www-anw.cs.umass.edu/~barto/courses/cs687/williams92simple.pdf)
#   Args:
#     logits_t: a sequence of unnormalized action preferences.
#     a_t: a sequence of actions sampled from the preferences `logits_t`.
#     adv_t: the observed or estimated advantages from executing actions `a_t`.
#     w_t: a per timestep weighting for the loss.
#     use_stop_gradient: bool indicating whether or not to apply stop gradient to
#       advantages.
#   Returns:
#     Loss whose gradient corresponds to a policy gradient update.
#   """
#   chex.assert_rank([logits_t, a_t, adv_t, w_t], [2, 1, 1, 1])
#   chex.assert_type([logits_t, a_t, adv_t, w_t], [float, int, float, float])
#
#   log_pi_a_t = distributions.softmax().logprob(a_t, logits_t)
#   adv_t = jax.lax.select(use_stop_gradient, jax.lax.stop_gradient(adv_t), adv_t)
#   loss_per_timestep = -log_pi_a_t * adv_t
#   return jnp.mean(loss_per_timestep * w_t)


def _compute_advantages(logits_t: Array,
                        q_t: Array,
                        use_stop_gradient=True) -> Array:
  """Computes summed advantage using logits and action values."""
  policy_t = jax.nn.softmax(logits_t, axis=1)

  # Avoid computing gradients for action_values.
  q_t = jax.lax.select(use_stop_gradient, jax.lax.stop_gradient(q_t), q_t)
  baseline_t = jnp.sum(policy_t * q_t, axis=1)

  adv_t = q_t - jnp.expand_dims(baseline_t, 1)
  return policy_t


def qpg_loss(
    logits_t: Array,
    q_t: Array,
    use_stop_gradient: bool = True,
) -> Array:
  """Computes the QPG (Q-based Policy Gradient) loss.
  See "Actor-Critic Policy Optimization in Partially Observable Multiagent
  Environments" by Srinivasan, Lanctot (https://arxiv.org/abs/1810.09026).
  Args:
    logits_t: a sequence of unnormalized action preferences.
    q_t: the observed or estimated action value from executing actions `a_t` at
      time t.
    use_stop_gradient: bool indicating whether or not to apply stop gradient to
      advantages.
  Returns:
    QPG Loss.
  """
  chex.assert_rank([logits_t, q_t], 2)
  chex.assert_type([logits_t, q_t], float)

  policy_t, advantage_t = _compute_advantages(logits_t, q_t)
  advantage_t = jax.lax.select(use_stop_gradient,
                               jax.lax.stop_gradient(advantage_t), advantage_t)
  policy_advantages = -policy_t * advantage_t
  loss = jnp.mean(jnp.sum(policy_advantages, axis=1), axis=0)
  return loss


def clipped_surrogate_pg_loss(
    prob_ratios_t: Array,
    adv_t: Array,
    epsilon: Scalar,
    use_stop_gradient=True) -> Array:
  """Computes the clipped surrogate policy gradient loss.
  L_clipₜ(θ) = - min(rₜ(θ)Âₜ, clip(rₜ(θ), 1-ε, 1+ε)Âₜ)
  Where rₜ(θ) = π_θ(aₜ| sₜ) / π_θ_old(aₜ| sₜ) and Âₜ are the advantages.
  See Proximal Policy Optimization Algorithms, Schulman et al.:
  https://arxiv.org/abs/1707.06347
  Args:
    prob_ratios_t: Ratio of action probabilities for actions a_t:
        rₜ(θ) = π_θ(aₜ| sₜ) / π_θ_old(aₜ| sₜ)
    adv_t: the observed or estimated advantages from executing actions a_t.
    epsilon: Scalar value corresponding to how much to clip the objecctive.
    use_stop_gradient: bool indicating whether or not to apply stop gradient to
      advantages.
  Returns:
    Loss whose gradient corresponds to a clipped surrogate policy gradient
        update.
  """
  chex.assert_rank([prob_ratios_t, adv_t], [1, 1])
  chex.assert_type([prob_ratios_t, adv_t], [float, float])

  adv_t = jax.lax.select(use_stop_gradient, jax.lax.stop_gradient(adv_t), adv_t)
  clipped_ratios_t = jnp.clip(prob_ratios_t, 1. - epsilon, 1. + epsilon)
  clipped_objective = jnp.fmin(prob_ratios_t * adv_t, clipped_ratios_t * adv_t)
  return -jnp.mean(clipped_objective)