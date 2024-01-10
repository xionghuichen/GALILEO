# Created by xionghuichen at 2022/10/25
# Email: chenxh@lamda.nju.edu.cn

from GALILEO.nets.net import *
from GALILEO.learner.base import BaseLearner
from GALILEO.losses.base import *
from GALILEO.utils import *
from RLA import exp_manager
from GALILEO.config import *
import tensorflow as tf

class Galileo(BaseLearner):
    def __init__(self, dm_model, pi_model, old_dm_model, old_pi_model, v_models, sas_dis_model, sa_dis_model,
                 model_opt, other_opt, model_lr, lr_v, lr_dis, other_lr, gamma, dataset_holder, sess,
                 trpo_only_for_model, rescale_grad, log_prob_clip=10, entcoeff=1e-3, d_entcoeff=1e-3, lam=0.95, l2_coeff=1e-6,
                 wbc_ent_coeff=5):
        super(Galileo, self).__init__(dataset_holder)
        self.dm_model = dm_model
        self.pi_model = pi_model
        self.log_prob_clip = log_prob_clip
        self.trpo_only_for_model = trpo_only_for_model
        self.old_dm_model = old_dm_model
        self.old_pi_model = old_pi_model
        self.sas_dis_model = sas_dis_model
        self.sa_dis_model = sa_dis_model
        self.sas_dm_v_model = v_models['sas_dm']
        self.sa_dm_v_model = v_models['sa_dm']
        self.sas_pi_v_model = v_models['sas_pi']
        self.sa_pi_v_model = v_models['sa_pi']
        self.wbc_ent_coeff = wbc_ent_coeff
        self.model_opt = model_opt
        self.other_opt = other_opt
        self.model_lr = model_lr
        self.other_lr = other_lr
        self.lr_v = lr_v
        self.lr_dis = lr_dis
        self.entcoeff = entcoeff
        self.d_entcoeff = d_entcoeff
        self.l2_coeff = l2_coeff
        self.rescale_grad = rescale_grad
        self.gamma = gamma
        self.lam = lam
        self.sess = sess
        self.init_extra_place_holder()

    def init_extra_place_holder(self):
        self.gen_s_input_ph = tf.placeholder(dtype=tf.float32, shape=[None, self.s_dim], name='gen_s')
        self.gen_a_input_ph = tf.placeholder(dtype=tf.float32, shape=[None, self.n_actions], name='gen_a')
        self.gen_p_s_next_target_ph = tf.placeholder(dtype=tf.float32, shape=[None, self.ps_dim], name='gen_ps')
        self.g_norm_ph = tf.placeholder(dtype=tf.float32, shape=None, name='g_norm')
        self.ret_sas_ph = tf.placeholder(dtype=tf.float32, shape=[None, 2], name='ret_sas')
        self.ret_sa_ph = tf.placeholder(dtype=tf.float32, shape=[None, 2], name='ret_sa')
        self.atarg_ph = tf.placeholder(dtype=tf.float32, shape=[None, 2])  # Target advantage function (if applicable)
        self.flat_tangent_ph = tf.placeholder(dtype=tf.float32, shape=[None], name="flat_tan")

    def graph_construction(self, scope='galileo', reuse=False):
        assert isinstance(self.dm_model, DM)
        assert isinstance(self.pi_model, Pi)
        assert isinstance(self.old_dm_model, DM)
        assert isinstance(self.old_pi_model, Pi)
        assert isinstance(self.sas_dis_model, Discriminator)
        assert isinstance(self.sa_dis_model, Discriminator)
        with tf.variable_scope(scope, reuse=reuse):
            # assert isinstance(self.v_model, V)
            norm_s_input = self.norm_s(self.s_input_ph)
            norm_a_input = self.norm_a(self.a_input_ph)
            norm_p_s_next_target = self.norm_p_s_next(self.p_s_next_target_ph)
            norm_gen_s_input = self.norm_s(self.gen_s_input_ph)
            norm_gen_a_input = self.norm_a(self.gen_a_input_ph)
            norm_gen_p_s_next_target = self.norm_p_s_next(self.gen_p_s_next_target_ph)
            # discriminator related variables construction
            logits_real_sas = self.sas_dis_model.logits((norm_s_input, norm_a_input, norm_p_s_next_target),
                                                        with_noise=True)
            logits_real_sa = self.sa_dis_model.logits((norm_s_input, norm_a_input, norm_p_s_next_target),
                                                      with_noise=True)
            logits_fake_sas = self.sas_dis_model.logits((norm_gen_s_input, norm_gen_a_input, norm_gen_p_s_next_target),
                                                        with_noise=True)
            logits_fake_sa = self.sa_dis_model.logits((norm_gen_s_input, norm_gen_a_input, norm_gen_p_s_next_target),
                                                      with_noise=True)
            loss_fake_sas = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits_fake_sas,
                                                                    labels=tf.zeros_like(logits_fake_sas))
            loss_real_sas = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits_real_sas,
                                                                    labels=tf.ones_like(logits_real_sas))
            self.prob_fake_sas = self.sas_dis_model.prob((norm_gen_s_input, norm_gen_a_input, norm_gen_p_s_next_target),
                                                         with_noise=False)
            self.prob_real_sas = self.sas_dis_model.prob((norm_s_input, norm_a_input, norm_p_s_next_target),
                                                         with_noise=False)
            self.prob_fake_sas_noise = tf.nn.sigmoid(logits_fake_sas)
            self.prob_fake_sa_noise = tf.nn.sigmoid(logits_fake_sa)
            logits_sas = tf.concat([logits_fake_sas, logits_real_sas], 0)
            loss_fake_sa = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits_fake_sa,
                                                                   labels=tf.zeros_like(logits_fake_sa))
            loss_real_sa = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits_real_sa,
                                                                   labels=tf.ones_like(logits_real_sa))
            self.prob_fake_sa = self.sa_dis_model.prob((norm_gen_s_input, norm_gen_a_input, norm_gen_p_s_next_target),
                                                       with_noise=False)
            self.prob_real_sa = self.sa_dis_model.prob((norm_s_input, norm_a_input, norm_p_s_next_target),
                                                       with_noise=False)
            self.prob_real_sas_noise = tf.nn.sigmoid(logits_real_sas)
            self.prob_real_sa_noise = tf.nn.sigmoid(logits_real_sa)
            self.r_sas_gen = self.sas_dis_model.reward((norm_gen_s_input, norm_gen_a_input, norm_gen_p_s_next_target),
                                                       with_noise=False)
            self.r_sa_gen = self.sa_dis_model.reward((norm_gen_s_input, norm_gen_a_input, norm_gen_p_s_next_target),
                                                     with_noise=False)
            logits_sa = tf.concat([logits_fake_sa, logits_real_sa], 0)
            # policy-dynamics related variable contruction
            self.dm_dist, self.dm_mean, self.dm_std = self.dm_model.obj_graph_construct(
                (norm_gen_s_input, norm_gen_a_input))
            self.dm_sample = self.dm_dist.sample()
            self.pi_dist, self.pi_mean, self.pi_std = self.pi_model.obj_graph_construct((norm_gen_s_input))
            self.dm_dist_real_data, self.dm_mean_real_data, self.dm_std_real_data = self.dm_model.obj_graph_construct(
                (norm_s_input, norm_a_input))
            self.pi_dist_real_data, self.pi_mean_real_data, self.pi_std_real_data = self.pi_model.obj_graph_construct(
                (norm_s_input))

            old_dm_dist, old_dm_mean, old_dm_std = self.old_dm_model.obj_graph_construct(
                (norm_gen_s_input, norm_gen_a_input))
            old_pi_dist, old_pi_mean, old_pi_std = self.old_pi_model.obj_graph_construct((norm_gen_s_input))
            # dm_dist, dm_mean, dm_std = self.dm_model.obj_graph_construct((norm_gen_s_input, norm_gen_a_input))
            # pi_dist, pi_mean, pi_std = self.pi_model.obj_graph_construct((norm_gen_s_input))

            old_dm_dist_real_data, _, _ = self.old_dm_model.obj_graph_construct((norm_s_input, norm_a_input))
            entropy_real_data = old_dm_dist_real_data.scale * self.wbc_ent_coeff  # H_M* is unknown in prior.

            dm_kloldnew = old_dm_dist.kl_divergence(self.dm_dist)
            dm_ent = self.dm_dist.entropy()
            dm_meankl = tf.reduce_mean(dm_kloldnew)
            dm_meanent = tf.reduce_mean(dm_ent)
            dm_entbonus = self.entcoeff * dm_meanent
            pi_kloldnew = old_pi_dist.kl_divergence(self.pi_dist)
            pi_ent = self.pi_dist.entropy()
            pi_meankl = tf.reduce_mean(pi_kloldnew)
            pi_meanent = tf.reduce_mean(pi_ent)
            pi_entbonus = self.entcoeff * pi_meanent
            meankl = pi_meankl + dm_meankl
            meanent = pi_meanent + dm_meanent
            entbonus = pi_entbonus + dm_entbonus

            dm_ratio = tf.exp(
                tf.reduce_sum(tf.log(self.dm_dist.prob(norm_gen_p_s_next_target) + EPSILON), axis=-1, keepdims=True)
                - tf.reduce_sum(tf.log(old_dm_dist.prob(norm_gen_p_s_next_target) + EPSILON), axis=-1,
                                keepdims=True))  # advantage * pnew / pold
            pi_ratio = tf.exp(
                tf.reduce_sum(tf.log(self.pi_dist.prob(norm_gen_a_input) + EPSILON), axis=-1, keepdims=True)
                - tf.reduce_sum(tf.log(old_pi_dist.prob(norm_gen_a_input) + EPSILON), axis=-1,
                                keepdims=True))  # advantage * pnew / pold

            ratio = tf.concat([pi_ratio, dm_ratio], axis=-1)

            # value related variable construction
            def v_construct(s, a):
                sa_input = tf.concat([s, a], axis=-1)
                v_sas = tf.concat([self.sas_pi_v_model.obj_graph_construct(sa_input),
                                   self.sas_dm_v_model.obj_graph_construct(sa_input)], axis=-1)
                v_sa = tf.concat([self.sa_pi_v_model.obj_graph_construct(s),
                                  self.sa_dm_v_model.obj_graph_construct(s)], axis=-1)
                return v_sas, v_sa

            self.v_sas_op, self.v_sa_op = v_construct(norm_gen_s_input, norm_gen_a_input)
            # end-to-end rollout
            norm_gen_a_sample = self.pi_dist.sample()
            self.gen_a_sample = self.denorm_a(norm_gen_a_sample)
            p_s_next_rollout_dist, _, _ = self.dm_model.obj_graph_construct((norm_gen_s_input, norm_gen_a_sample))
            norm_p_s_next_rollout = p_s_next_rollout_dist.sample()
            self.p_s_next_rollout = self.denorm_p_s_next(norm_p_s_next_rollout)
            self.r_sas_rollout = self.sas_dis_model.reward((norm_gen_s_input, norm_gen_a_sample, norm_p_s_next_rollout),
                                                           with_noise=False)
            self.r_sa_rollout = self.sa_dis_model.reward((norm_gen_s_input, norm_gen_a_sample, norm_p_s_next_rollout),
                                                         with_noise=False)
            self.v_sas_rollout, self.v_sa_rollout = v_construct(norm_gen_s_input, norm_gen_a_sample)

            # --- compute losses ----
            # 1. discrimnator losses
            entropy = tf.reduce_mean(logit_bernoulli_entropy(logits_sa))
            self.dis_loss_sa = tf.reduce_mean(loss_fake_sa) + tf.reduce_mean(loss_real_sa) + entropy * self.d_entcoeff
            entropy_sas = tf.reduce_mean(logit_bernoulli_entropy(logits_sas))
            self.dis_loss_sas = tf.reduce_mean(loss_fake_sas) + tf.reduce_mean(
                loss_real_sas) + entropy_sas * self.d_entcoeff
            # 2. policy-dynamics model losses
            model_neg_loglikelihood = neg_loglikelihoood(self.dm_dist_real_data, norm_p_s_next_target,
                                                         self.log_prob_clip)
            # - negative weight makes gradient unstable -> only keep the positive parts.
            clip_weight = tf.stop_gradient(entropy_real_data + tf.clip_by_value(- tf.expand_dims(self.atarg_ph[..., -1], axis=-1), 0, 1))
            self.v_weight_model_likelihood_loss = tf.reduce_mean(clip_weight * model_neg_loglikelihood)
            # - loglikelhood
            self.model_lilelihood_loss = tf.reduce_mean(model_neg_loglikelihood)
            self.pi_likelihood_loss = tf.reduce_mean(neg_loglikelihoood(self.pi_dist_real_data, norm_a_input))
            # - trpo loss
            surrgain = tf.reduce_mean(ratio * self.atarg_ph)
            self.optimgain = surrgain + entbonus
            if self.trpo_only_for_model:
                trpo_var_list = self.dm_model.trainable_variables()
            else:
                trpo_var_list = self.dm_model.trainable_variables() + self.pi_model.trainable_variables()
            l2_loss = tf.add_n([tf.nn.l2_loss(l2_var) for l2_var in trpo_var_list]) / len(trpo_var_list)
            self.optimgain -= self.l2_coeff * l2_loss
            mean_ratio = tf.reduce_mean(ratio)
            self.losses = [self.optimgain, meankl, entbonus, surrgain, meanent, mean_ratio, l2_loss]
            self.loss_names = ["optimgain", "meankl", "entloss", "surrgain", "entropy", 'mean_ratio', 'l2_loss']
            shapes = [var.get_shape().as_list() for var in trpo_var_list]
            self.policy_gradient = flatgrad(self.optimgain, trpo_var_list)
            self._set_from_flat, self._set_from_flat_phs = set_from_flat(trpo_var_list)
            self.get_flat = get_flat(trpo_var_list)

            start = 0
            tangents = []
            klgrads = tf.gradients(meankl, trpo_var_list)
            for shape in shapes:
                sz = intprod(shape)
                tangents.append(tf.reshape(self.flat_tangent_ph[start:start + sz], shape))
                start += sz
            gvp = tf.add_n(
                [tf.reduce_sum(g * tangent) for (g, tangent) in zipsame(klgrads, tangents)])  # pylint: disable=E1111
            self.fvp = flatgrad(gvp, trpo_var_list)
            # compute_fvp = U.function([flat_tangent_ph, gen_s_input, gen_a_input, atarg_ph], fvp)
            self.assign_vars = [tf.assign(oldv, newv) for (oldv, newv) in
                                zipsame(self.old_dm_model.trainable_variables(),
                                        self.dm_model.trainable_variables())] + \
                               [tf.assign(oldv, newv) for (oldv, newv) in
                                zipsame(self.old_pi_model.trainable_variables(),
                                        self.pi_model.trainable_variables())]
            # - v loss
            self.vferr_sas = tf.reduce_mean(tf.square(self.v_sas_op - self.ret_sas_ph))
            self.vferr_sa = tf.reduce_mean(tf.square(self.v_sa_op - self.ret_sa_ph))
            # --- construct opt ---
            # 1. discriminator
            dis_sas_optimizer = tf.train.RMSPropOptimizer(learning_rate=self.lr_dis, name='dis_sas_opt')
            dis_sa_optimizer = tf.train.RMSPropOptimizer(learning_rate=self.lr_dis, name='dis_sa_opt')
            self.dis_sas_opt_op = dis_sas_optimizer.minimize(self.dis_loss_sas,
                                                             var_list=self.sas_dis_model.trainable_variables())
            self.dis_sa_opt_op = dis_sa_optimizer.minimize(self.dis_loss_sa,
                                                           var_list=self.sa_dis_model.trainable_variables())

            # 2. policy-dynamics
            model_opt = self.model_opt(learning_rate=self.model_lr, name='model_lr')
            grads_sl, variables_sl = zip(*model_opt.compute_gradients(self.v_weight_model_likelihood_loss +
                                                                      self.l2_coeff * l2_loss,
                                                                      var_list=self.dm_model.trainable_variables()))
            sl_grad, self.sl_global_norm = tf.clip_by_global_norm(grads_sl, self.g_norm_ph)
            self.v_weight_likelihood_op = model_opt.apply_gradients(
                grads_and_vars=[(g, v) for g, v in zip(sl_grad, variables_sl)])
            policy_opt = self.other_opt(learning_rate=self.other_lr, name='pi_opt')
            self.pi_likelihood_op = policy_opt.minimize(self.pi_likelihood_loss + self.l2_coeff * l2_loss,
                                                        var_list=self.pi_model.trainable_variables())
            sl_grad_no_v, variables_sl = zip(
                *model_opt.compute_gradients(self.model_lilelihood_loss + self.l2_coeff * l2_loss,
                                             var_list=self.dm_model.trainable_variables()))
            sl_grad_no_v, self.sl_global_no_v_norm = tf.clip_by_global_norm(sl_grad_no_v, self.g_norm_ph)
            if self.rescale_grad:
                self.model_likelihood_op = model_opt.apply_gradients(
                    grads_and_vars=[(g, v) for g, v in zip(sl_grad_no_v, variables_sl)])
            else:
                self.model_likelihood_op = model_opt.minimize(self.model_lilelihood_loss + self.l2_coeff * l2_loss)
            # 3. value
            v_sas_optimizer = self.other_opt(learning_rate=self.lr_v, name='v_op_sas')
            v_sa_optimizer = self.other_opt(learning_rate=self.lr_v, name='v_op_sa')
            self.v_sas_opt = v_sas_optimizer.minimize(self.vferr_sas,
                                                      var_list=self.sas_pi_v_model.trainable_variables() +
                                                               self.sas_dm_v_model.trainable_variables())
            self.v_sa_opt = v_sa_optimizer.minimize(self.vferr_sa,
                                                    var_list=self.sa_pi_v_model.trainable_variables() +
                                                             self.sa_dm_v_model.trainable_variables())

    def set_from_flat(self, vars):
        self.sess.run(self._set_from_flat, feed_dict={self._set_from_flat_phs: vars})

    def rollout_samples(self, S_init, env, traj_num, horizon, terminal_fn):
        # TODO: single dis test
        sample_idx = np.random.randint(0, S_init.shape[0], traj_num)
        s = S_init[sample_idx]
        S_train = []
        A_train = []
        R_SAS_train = []
        R_SA_train = []
        P_S_next_train = []
        V_SAS_train = []
        V_SA_train = []
        S_next_train = []
        D_train = []
        to_remove = []
        have_done = np.zeros((s.shape[0], 1), dtype=np.bool)
        r_sas = v_sas = None
        exp_manager.time_record('rollout_sample')
        for i in range(int(horizon)):
            a, p_s_next, r_sas, r_sa, v_sas, v_sa = self.sess.run(
                [self.gen_a_sample, self.p_s_next_rollout, self.r_sas_rollout, self.r_sa_rollout,
                 self.v_sas_rollout, self.v_sa_rollout], feed_dict={self.gen_s_input_ph: s})
            s_next = env.complete_env(s, a, p_s_next)
            d = terminal_fn(s, a, s_next)
            to_remove.append(have_done)
            have_done = have_done | d
            S_train.append(s)
            A_train.append(a)
            R_SAS_train.append(r_sas)
            R_SA_train.append(r_sa)
            V_SAS_train.append(v_sas)
            V_SA_train.append(v_sa)
            P_S_next_train.append(p_s_next)
            S_next_train.append(s_next)
            D_train.append(have_done)
            s = s_next
            if np.mean(have_done) == 1:
                break
            pass

        exp_manager.time_record_end('rollout_sample')
        D_train[-1] = np.ones(r_sas.shape)
        seg = {
            "S": S_train,
            "A": A_train,
            "R_SAS": R_SAS_train,
            "R_SA": R_SA_train,
            "D": D_train,
            "V_SAS": V_SAS_train,
            "V_SA": V_SA_train,
            "P_S_next": P_S_next_train,
            # "MSE_REW_train": MSE_REW_train,
            "V_next": np.zeros(v_sas.shape),  # the last step must be a done step
        }

        exp_manager.time_record('rollout_adv')
        add_vtarg_and_adv(seg, self.gamma, self.lam, i + 1, traj_num, adv_type='sa')
        add_vtarg_and_adv(seg, self.gamma, self.lam, i + 1, traj_num, adv_type='sas')
        add_vtarg_and_adv(seg, self.gamma, self.lam, i + 1, traj_num, adv_type='mixed')
        exp_manager.time_record_end('rollout_adv')
        # TODO: 重新测试新版
        # seg['adv_mixed'] = seg['tdlamret_sas'] - seg['tdlamret_sa']
        exp_manager.time_record('rollout_rm')
        remove_masked_data(seg, to_remove)
        exp_manager.time_record_end('rollout_rm')
        return seg

    def evaluate_next_state(self, s, a, deter=True):
        if deter:
            norm_ps_next = self.sess.run(self.dm_mean, feed_dict={self.gen_s_input_ph: s, self.gen_a_input_ph: a})
        else:
            norm_ps_next = self.sess.run(self.dm_sample, feed_dict={self.gen_s_input_ph: s, self.gen_a_input_ph: a})
        return self.denorm_p_s_next(norm_ps_next)

    def dis_data(self, real_data_s, real_data_a, real_data_ns, fake_data_s, fake_data_a, fake_data_ns):
        res_fake, res_real = self.sess.run([self.prob_fake_sas, self.prob_real_sas],
                                           feed_dict={self.s_input_ph: real_data_s,
                                                      self.a_input_ph: real_data_a,
                                                      self.p_s_next_target_ph: real_data_ns,
                                                      self.gen_s_input_ph: fake_data_s,
                                                      self.gen_a_input_ph: fake_data_a,
                                                      self.gen_p_s_next_target_ph: fake_data_ns})
        return res_fake, res_real