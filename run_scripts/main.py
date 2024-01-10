# Created by xionghuichen at 2022/7/26
# Email: chenxh@lamda.nju.edu.cn

from RLA import logger, exp_manager, ExperimentLoader
from RLA.rla_argparser import arg_parser_postprocess, boolean_flag
from baselines.common import set_global_seeds, tf_util as U
import argparse
import os
import os.path as osp
from collections import deque

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
from GALILEO.utils import *
from GALILEO.config import *
from auto_config_map import *
from GALILEO.offline_data.dataset_handler import DatasetHandler
from GALILEO.offline_data.dataloader import *
from GALILEO.losses.base import *
from GALILEO.envs.term_fn import is_terminal
from GALILEO.learner.bc import BC
from GALILEO.learner.ipw import IPW
from GALILEO.learner.galileo import Galileo
from GALILEO.nets.net import DM, Discriminator, V, Pi

from GALILEO.evaluation.dope_policy import D4RLPolicy
from GALILEO.envs.gnfc import GnfcEnv
from GALILEO.envs.tcga import TcgaEnv
from GALILEO.envs.mujoco import MjEnv
from GALILEO.learner.dynamics_model import DMEnv
from GALILEO.evaluation.func import *

def argsparser():
    parser = argparse.ArgumentParser("Train coupon policy in simulator")
    parser.add_argument('--seed', type=int, default=6)
    parser.add_argument("--data_type", default=DataType.D4RL, type=str)
    parser.add_argument("--horizon", default=50, type=int)
    parser.add_argument("--iters", default=50000, type=int)
    # for d4rl
    parser.add_argument("--env_name", default='walker2d', type=str)
    parser.add_argument("--data_train_type", default='medium', type=str)
    parser.add_argument('--load_eval_policy_path', type=str, default='../data/dope_policy/')
    parser.add_argument('--num_eval_episodes', type=int, default=10)
    parser.add_argument('--device', type=int, default=0)
    # for tcga
    parser.add_argument("--select_treatment", default=2, type=int)
    parser.add_argument("--treatment_selection_bias", default=2.0, type=float)
    parser.add_argument("--dosage_selection_bias", default=2.0, type=float)
    # for gnfc
    parser.add_argument('--dim', type=int, default=2)
    parser.add_argument('--alpha', type=float, default=0.8)
    parser.add_argument('--dm_noise', type=float, default=2.0)
    parser.add_argument('--noise_scale', type=float, default=0.2)
    parser.add_argument('--random_prob', type=float, default=0.05)
    parser.add_argument('--data_seed', type=int, default=9)
    boolean_flag(parser, 'one_step_dyn', default=False)
    # for rla
    parser.add_argument('--info', default='default exp info', type=str)
    parser.add_argument('--loaded_date', default=None, type=str)
    parser.add_argument('--loaded_task_name', default=None, type=str)
    # for galileo
    parser.add_argument('--alg_type', default=AlgType.GALILEO, type=str)
    parser.add_argument('--lr', default=0.0001, type=float)
    parser.add_argument('--lr_v', default=0.0001, type=float)
    parser.add_argument('--lr_dis', default=0.001, type=float)
    parser.add_argument('--std_bound', default=0.005, type=float)
    parser.add_argument('--dis_noise', default=0.005, type=float)
    parser.add_argument('--occ_noise_coef', default=1.0, type=float)
    parser.add_argument('--lr_dm', type=float, default=0.0003)
    parser.add_argument('--hid_dim', default=256, type=int)
    parser.add_argument("--g_step", default=2, type=int)  # TODO: 2 only tested in mujoco. we should check the performance in other tasks firstly.
    parser.add_argument("--bc_step", default=4, type=int)
    parser.add_argument('--bc_batch_size', type=int, default=50000)
    parser.add_argument('--sample_traj_size', type=int, default=400)
    # both d_step and dis_batch_traj_size should be large enough to cover enough data for updating.
    parser.add_argument("--d_step", default=1, type=int)
    parser.add_argument('--dis_batch_traj_size', type=int, default=400)
    parser.add_argument('--dis_mini_batch_num', type=int, default=10)
    parser.add_argument('--mse_relax_coef', type=int, default=0.1)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--max_kl', type=float, default=0.0005)
    parser.add_argument('--pi_pretrain', type=int, default=10)
    # set as default
    parser.add_argument('--cg_damping',  type=float, default=1e-2)
    parser.add_argument('--cg_iters',  type=int, default=10)
    parser.add_argument('--vf_iters',  type=int, default=4)
    parser.add_argument('--dm_std_init',  type=float, default=0.3)
    boolean_flag(parser, 'auto_d_noise', default=True)
    # experiment
    boolean_flag(parser, 'trpo_only_for_model', default=False)  # only for model 效果不大行
    boolean_flag(parser, 'only_model_likelihood', default=False)
    boolean_flag(parser, 'rescale_grad', default=True)
    boolean_flag(parser, 's_add_noise', default=True)
    boolean_flag(parser, 'debug', default=False)
    boolean_flag(parser, 'res_dis', default=False)
    boolean_flag(parser, 'zero_mean', default=True)
    boolean_flag(parser, 'atten_dis', default=False)
    parser.add_argument('--l2_coeff',  type=float, default=1e-6)
    parser.add_argument('--d_entcoeff',  type=float, default=1e-3)
    # 这个值不能调低，会让dis prob 变差。我们需要持续依赖于似然概率，来保证获得的模型在似然含义上是准确的，然后再考虑对抗调优的问题
    parser.add_argument('--log_prob_clip',  type=float, default=10.)
    args = parser.parse_args()
    return args


def main():
    args = argsparser()
    set_global_seeds(args.data_seed)
    if args.horizon == 1000: # reduce the memory used
        args.sample_traj_size = 20
        args.dis_batch_traj_size = 20
        args.dis_mini_batch_num = 3

    def get_package_path():
        return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    kwargs = vars(args)
    # load hyper-parameters
    if args.loaded_task_name != '':
        exp_manager.set_hyper_param(**kwargs)
        exp_loader = ExperimentLoader()
        exp_loader.config(task_name=args.loaded_task_name, record_date=args.loaded_date, root=get_package_path())
        args = exp_loader.import_hyper_parameters()
        kwargs = vars(args)
    # hyper-parameter map
    update_hp(kwargs)

    exp_manager.set_hyper_param(**kwargs)
    args = argparse.Namespace(**kwargs)
    # rla config

    record_param = ['info', 'seed', 'data_type']
    if args.data_type == DataType.D4RL:
        record_param += ['env_name', 'data_train_type']
    elif args.data_type == DataType.TCGA:
        record_param += ['select_treatment', 'dosage_selection_bias']
    elif args.data_type == DataType.GNFC:
        record_param += ['random_prob', 'noise_scale']
    extra_interested_param = ['horizon', 'max_kl', 'alg_type', 'log_prob_clip', 'std_bound', 'occ_noise_coef', 'max_kl']
    exp_manager.add_record_param(record_param + extra_interested_param)
    if args.debug:
        task_table_name = 'debug'
    else:
        task_table_name = args.data_type + '-v4'
    exp_manager.configure(task_table_name=task_table_name,
                          rla_config=os.path.join(get_package_path(), 'rla_config.yaml'),
                          ignore_file_path=os.path.join(get_package_path(), '.gitignore'),
                          data_root=get_package_path())
    exp_manager.log_files_gen()
    if args.loaded_task_name != '':
        #     exp_loader = ExperimentLoader()
        exp_loader.fork_log_files()
    # dataset generation
    set_global_seeds(args.data_seed)

    if args.data_type == DataType.D4RL:
        dataset = D4rlDataset(env_name=args.env_name, horizon=args.horizon, train_type=args.data_train_type)
        dope_name = args.env_name
        if args.env_name == 'halfcheetah':
            generate = 'HalfCheetah-v2'
        elif args.env_name == 'walker2d':
            generate = 'Walker2d-v2'
            dope_name = 'walker'
        elif args.env_name == 'hopper':
            generate = 'Hopper-v2'
        else:
            raise NotImplementedError
        env = gym.make(generate)
        env = MjEnv(env)
        if args.device == -1:
            device = 'cpu'
        else:
            device = 'cuda:'+str(args.device)
        eval_policy_set = [D4RLPolicy(osp.join(args.load_eval_policy_path, args.env_name, f"{dope_name}_online_{idx}.pkl"), device=device) for idx in
            range(10)]

    elif args.data_type == DataType.TCGA:
        dataset = TcgaDataset(args.select_treatment, args.treatment_selection_bias, args.dosage_selection_bias,
                              data_location=osp.join(get_package_path(), 'datasets/tcga.p'))
        env = TcgaEnv(dataset)
    elif args.data_type == DataType.GNFC:
        target_line = MAX_S_MEAN / args.alpha
        dataset = GnfcDataset(args.dim, args.dm_noise, args.one_step_dyn, target_line,
                              args.random_prob, args.noise_scale, args.horizon, args.data_seed,
                              dataset_traj_size=100000)
        env = GnfcEnv(args.dim, args.dm_noise, args.one_step_dyn, dataset)
    else:
        raise NotImplementedError

    data_handler = DatasetHandler(dataset=dataset, data_type=args.data_type, env=env)
    terminal_fn = lambda s, a, ns: is_terminal(s, a, ns, args.env_name)
    set_global_seeds(args.seed)
    n_actions = data_handler.a_dim
    dim_next = len(data_handler.p_s_n_mean)
    sess = U.make_session(make_default=True)
    dm_model = DM(args.hid_dim, dim_next, args.dm_std_init)
    model_opt = tf.train.RMSPropOptimizer
    other_opt = tf.train.AdamOptimizer
    # construct graphs
    if args.alg_type == AlgType.SL:
        bc = BC(dm_model, model_opt, args.lr_dm, data_handler, sess)
        bc.graph_construction()
    elif args.alg_type == AlgType.IPW:
        pi_model = Pi(args.std_bound, n_actions, args.hid_dim)
        ipw = IPW(dm_model, pi_model, model_opt, other_opt, args.lr_dm, args.lr, data_handler, sess)
        ipw.graph_construction()
    elif args.alg_type == AlgType.GALILEO or args.alg_type == AlgType.GAIL:
        old_dm_model = DM(args.hid_dim, dim_next, args.dm_std_init, 'old_dm')
        pi_model = Pi(args.std_bound, n_actions, args.hid_dim)
        old_pi_model = Pi(args.std_bound, n_actions, args.hid_dim, scope='old_pi')
        v_models = {
            "sas_dm": V(args.hid_dim, 'sas_dm'),
            "sa_dm": V(args.hid_dim, 'sa_dm'),
            "sas_pi": V(args.hid_dim, 'sas_pi'),
            "sa_pi": V(args.hid_dim, 'sa_pi'),
        }
        sas_dis_model = Discriminator(args.hid_dim, args.dis_noise, mask_s_next=False, scope='sas_dis', s_add_noise=args.s_add_noise,
                                      res_dis=args.res_dis, occ_noise_coef=args.occ_noise_coef)
        sa_dis_model = Discriminator(args.hid_dim, args.dis_noise, mask_s_next=True, scope='sa_dis', s_add_noise=args.s_add_noise,
                                      res_dis=args.res_dis, occ_noise_coef=args.occ_noise_coef)
        galileo = Galileo(dm_model, pi_model, old_dm_model, old_pi_model, v_models, sas_dis_model, sa_dis_model,
                          model_opt, other_opt, args.lr_dm, args.lr_v, args.lr_dis, args.lr, args.gamma, dataset_holder=data_handler, sess=sess,
                          trpo_only_for_model=args.trpo_only_for_model, rescale_grad=args.rescale_grad, d_entcoeff=args.d_entcoeff,
                          l2_coeff=args.l2_coeff, log_prob_clip=args.log_prob_clip)
        galileo.graph_construction()
    else:
        raise NotImplementedError

    dm_env_evaluation = DMEnv(data_handler=data_handler, dm_model=dm_model, sess=sess, terminal_fn=terminal_fn,
                   gym_env=env.env, branch_init=False, episode_len=args.horizon,
                   deter_pred=True, use_real_env=args.alg_type == AlgType.REAL_ENV_MODEL_ROLLOUT,
                   real_reset=True, acs_cons_scale=-1, state_cons=False)
    dm_env_evaluation.graph_construction()
    # initialize tf variables
    sess.run(tf.initialize_all_variables())
    exp_manager.new_saver(max_to_keep=1)
    # initialize python variables
    idx_now = 0
    idx = np.repeat(np.arange(data_handler.A.shape[0]), 2)
    np.random.shuffle(idx)
    bc_bat = args.bc_batch_size
    # if args.loaded_task_name != '':
    exp_loader.load_from_record_date()
    if args.alg_type == AlgType.GALILEO:
        # pretrain pi.
        for i in range(args.pi_pretrain):
            exp_manager.time_step_holder.set_time(i)
            mse_idx = np.random.randint(0, data_handler.data_size, args.bc_batch_size)
            pi_likelihood, pi_std = sess.run([galileo.pi_likelihood_loss, galileo.pi_std, galileo.dm_std,
                                              galileo.pi_likelihood_op],
                                              feed_dict={galileo.s_input_ph: data_handler.S[mse_idx],
                                                         galileo.a_input_ph: data_handler.A[mse_idx],
                                                         galileo.p_s_next_target_ph: data_handler.P_S_next[mse_idx]
                                                         })[:-2]
            if i % 1000 == 0:
                logger.record_tabular(f'pretrain/pi_std_max', np.max(pi_std))
                logger.record_tabular(f'pretrain/pi_std_mean', np.mean(pi_std))
                logger.record_tabular(f'pretrain/pi_std_min', np.min(pi_std))
                logger.record_tabular(f'pretrain/pi_likelihood', np.mean(pi_likelihood))
                logger.dump_tabular()
        pass
    if args.alg_type == AlgType.IPW:
        best_mse = np.inf
        break_counter = 0
        for i in range(100000):
            exp_manager.time_step_holder.set_time(i)
            mse_idx = np.random.randint(0, data_handler.data_size, args.bc_batch_size)
            pi_likelihood, pi_mse, pi_std = sess.run([ipw.pi_likelihood_loss, ipw.mse_loss, ipw.pi_std, ipw.dm_std,
                                              ipw.pi_likelihood_op],
                                              feed_dict={ipw.s_input_ph: data_handler.S[mse_idx],
                                                         ipw.a_input_ph: data_handler.A[mse_idx],
                                                         ipw.p_s_next_target_ph: data_handler.P_S_next[mse_idx]
                                                         })[:-2]
            if i % 1000 == 0:
                logger.record_tabular(f'pretrain/pi_std_max', np.max(pi_std))
                logger.record_tabular(f'pretrain/pi_std_mean', np.mean(pi_std))
                logger.record_tabular(f'pretrain/pi_std_min', np.min(pi_std))
                logger.record_tabular(f'pretrain/pi_likelihood', np.mean(pi_likelihood))
                logger.record_tabular(f'pretrain/pi_mse', np.mean(pi_mse))
                logger.dump_tabular()
            # if i % 100 == 0:
            #     if best_mse > np.mean(pi_mse):
            #         best_mse = np.mean(pi_mse)
            #         break_counter = 0
            #     else:
            #         break_counter += 1
            #     if break_counter > 10:
            #         break

    real_values = None
    for i in range(args.iters):
        exp_manager.time_record('iter')
        exp_manager.time_step_holder.set_time(i)
        if args.alg_type == AlgType.SL:
            # exp_manager.time_record('train')

            for j in range(args.bc_step + args.g_step):
                if (idx_now + 1) * bc_bat > data_handler.A.shape[0]:
                    idx_now = 0
                mse_out, dm_std_out = sess.run([bc.loss, bc.dm_std, bc.train_op],
                                               feed_dict={bc.s_input_ph: data_handler.S[
                                                   idx[idx_now * bc_bat:(idx_now + 1) * bc_bat]],
                                                          bc.a_input_ph: data_handler.A[
                                                              idx[idx_now * bc_bat:(idx_now + 1) * bc_bat]],
                                                          bc.p_s_next_target_ph: data_handler.P_S_next[
                                                              idx[idx_now * bc_bat:
                                                                  (idx_now + 1) * bc_bat]]})[:-1]
                idx_now += 1
            # exp_manager.time_record_end('train')
            if i % 10 == 0:
                logger.record_tabular("loss/model_neg_log_likelihood", np.mean(mse_out))
                logger.record_tabular("trpo/dm_std", np.mean(dm_std_out))
            if i % 400 == 0 and i > 0:
                # exp_manager.time_record('evaluation')
                data_handler.evaluation(do_plot=True, predict_fn=bc.evaluate_next_state, dis_pred_fn=None)
                exp_manager.save_checkpoint()
                # exp_manager.time_record_end('evaluation')
            logger.dump_tabular()
        elif args.alg_type == AlgType.IPW:
            for j in range(args.bc_step + args.g_step):
                if (idx_now + 1) * bc_bat > data_handler.A.shape[0]:
                    idx_now = 0
                mse_out, dm_std_out = sess.run([ipw.mse_loss, ipw.dm_std, ipw.train_op],
                                               feed_dict={ipw.s_input_ph: data_handler.S[
                                                   idx[idx_now * bc_bat:(idx_now + 1) * bc_bat]],
                                                          ipw.a_input_ph: data_handler.A[
                                                              idx[idx_now * bc_bat:(idx_now + 1) * bc_bat]],
                                                          ipw.p_s_next_target_ph: data_handler.P_S_next[
                                                              idx[idx_now * bc_bat:
                                                                  (idx_now + 1) * bc_bat]]})[:-1]
                idx_now += 1
            # exp_manager.time_record_end('train')
            if i % 50 == 0:
                logger.record_tabular("loss/model_neg_log_likelihood", np.mean(mse_out))
                logger.record_tabular("trpo/dm_std", np.mean(dm_std_out))
                logger.dump_tabular()
            if i % 1000 == 0 and i > 0:
                # exp_manager.time_record('evaluation')
                data_handler.evaluation(do_plot=True, predict_fn=ipw.evaluate_next_state, dis_pred_fn=None)
                exp_manager.save_checkpoint()
                logger.dump_tabular()

        elif args.alg_type == AlgType.GALILEO or args.alg_type == AlgType.GAIL:
            recent_adv_max = deque(maxlen=100)
            recent_adv_min = deque(maxlen=100)
            exp_manager.time_record('G')
            target_max_kl = args.max_kl
            for gi in range(args.g_step):
                S_start = data_handler.traj_S[0]
                exp_manager.time_record('rollout')
                seg = galileo.rollout_samples(S_start, env, args.sample_traj_size, args.horizon, terminal_fn)
                avg_len = seg['A'].shape[0] / args.sample_traj_size
                atarg = seg['adv_mixed']  # (24)
                seg_a = seg['A']
                seg_s = seg['S']
                seg_next_s = seg['P_S_next']
                # if args.zero_mean:
                #     atarg = (atarg - atarg.mean(axis=0)) / atarg.std(axis=0)  # standardized advantage function estimate
                # else:
                atarg = (atarg) / atarg.std(axis=0)  # standardized advantage function estimate
                fvpargs = seg['S'], seg['A'], atarg
                fvpargs = [arr[::5] for arr in fvpargs]
                exp_manager.time_record_end('rollout')

                exp_manager.time_record('trpo')
                sess.run(galileo.assign_vars)
                exp_manager.time_record('computegrad')
                optmgain_before, g = sess.run([galileo.optimgain, galileo.policy_gradient], feed_dict={
                    galileo.atarg_ph: atarg,
                    galileo.gen_a_input_ph: seg_a,
                    galileo.gen_s_input_ph: seg_s,
                    galileo.gen_p_s_next_target_ph: seg_next_s,
                })
                exp_manager.time_record_end('computegrad')
                if np.allclose(g, 0):
                    logger.log("Got zero gradient. not updating")
                else:
                    def fisher_vector_product(p):
                        fvp_out = sess.run(galileo.fvp,
                                           feed_dict={galileo.flat_tangent_ph: p,
                                                      galileo.gen_s_input_ph: fvpargs[0],
                                                      galileo.gen_a_input_ph: fvpargs[1],
                                                      galileo.atarg_ph: fvpargs[2]})
                        return fvp_out + args.cg_damping * p

                    stepdir = cg(fisher_vector_product, g, cg_iters=args.cg_iters, verbose=True)
                    shs = .5 * stepdir.dot(fisher_vector_product(stepdir))
                    lm = np.sqrt(shs / target_max_kl)
                    fullstep = stepdir / lm
                    expectedimprove = g.dot(fullstep)
                    surrbefore = optmgain_before
                    stepsize = 1.0
                    thbefore = sess.run(galileo.get_flat)
                    for _ in range(10):
                        thnew = thbefore + fullstep * stepsize
                        galileo.set_from_flat(thnew)
                        meanlosses = surr, kl, *_ = sess.run(galileo.losses, feed_dict={
                            galileo.atarg_ph: atarg,
                            galileo.gen_a_input_ph: seg_a,
                            galileo.gen_s_input_ph: seg_s,
                            galileo.gen_p_s_next_target_ph: seg_next_s, })
                        improve = surr - surrbefore
                        logger.log("Expected: %.3f Actual: %.3f" % (expectedimprove, improve))
                        if not np.isfinite(meanlosses).all():
                            logger.log("Got non-finite value of losses -- bad!")
                        elif kl > target_max_kl * 1.5:
                            logger.log("violated KL constraint. shrinking step.")
                        elif improve < 0:
                            logger.log("surrogate didn't improve. shrinking step.")
                        else:
                            logger.log("Stepsize OK!")
                            break
                        stepsize *= .5
                    else:
                        logger.log("couldn't compute a good step")
                        galileo.set_from_flat(thbefore)
                exp_manager.time_record_end('trpo')
                exp_manager.time_record('V')
                for _ in range(args.vf_iters):
                    vout_sas, v_err_out_sas, vout_sa, v_err_out_sa = sess.run([galileo.v_sas_op, galileo.vferr_sas,
                                                                               galileo.v_sa_op, galileo.vferr_sa,
                                                                               galileo.v_sas_opt, galileo.v_sa_opt,
                                                                               ], feed_dict={
                        galileo.gen_s_input_ph: seg["S"],
                        galileo.gen_a_input_ph: seg["A"],
                        galileo.ret_sas_ph: seg["tdlamret_sas"],
                        galileo.ret_sa_ph: seg["tdlamret_sa"]
                    })[:-2]
                exp_manager.time_record_end('V')
            if args.g_step > 0:
                for k, v in zip(galileo.loss_names, meanlosses):
                    logger.record_tabular(f'trpo/{k}', v, freq=20)
                logger.record_tabular(f'grad/original_g_norm', np.linalg.norm(g), freq=20)
                logger.record_tabular(f'trpo/stepsize', stepsize)
                logger.record_tabular(f'trpo/kl', kl)
                logger.record_tabular(f'trpo/target_max_kl', target_max_kl)
                logger.record_tabular(f'trpo/improve', improve)
                logger.record_tabular(f'vf/vout_sas', np.mean(vout_sas), freq=20)
                logger.record_tabular(f'vf/vout_sa', np.mean(vout_sa), freq=20)
                logger.record_tabular(f'vf/v_err_out_sas', np.mean(v_err_out_sas), freq=20)
                logger.record_tabular(f'vf/v_err_out_sa', np.mean(v_err_out_sa), freq=20)
                logger.record_tabular(f'trpo/atarg', np.mean(atarg))
                logger.record_tabular(f'trpo/atarg_max', np.max(atarg))
                logger.record_tabular(f'trpo/atarg_min', np.min(atarg))
                logger.record_tabular(f'perf/avg_length', avg_len)
            exp_manager.time_record_end('G')
            exp_manager.time_record('BC')
            if args.only_model_likelihood:
                do_v_weight_mse = False
                g_norm = 1
            else:
                do_v_weight_mse = (0.5-args.mse_relax_coef < (1 - np.exp(- seg['R_SAS'])).mean() < 0.5+args.mse_relax_coef) and \
                                  (0.5-args.mse_relax_coef < (1 - np.exp(- seg['R_SA'])).mean() < 0.5+args.mse_relax_coef)
                if args.rescale_grad:
                    g_norm = np.linalg.norm(fullstep * stepsize)
                else:
                    g_norm = 1
            logger.record_tabular(f'grad/trpo_g_norm', g_norm, freq=20)
            for bi in range(args.bc_step):
                if do_v_weight_mse:
                    bc_traj_size = int(args.bc_batch_size / args.horizon) + 1
                    if bc_traj_size >= data_handler.data_traj_size:
                        # traj_idx = np.random.randint(0, data_handler.data_traj_size, bc_traj_size)
                        D_S = data_handler.traj_S.reshape([-1, data_handler.s_dim])
                        D_A = data_handler.traj_A.reshape([-1, data_handler.a_dim])
                        D_P_S_next = data_handler.traj_P_S_next.reshape([-1, data_handler.ps_dim])
                        D_masks = data_handler.traj_masks.reshape([-1, 1])
                        bc_traj_size = data_handler.data_traj_size
                    else:
                        traj_idx = np.random.randint(0, data_handler.data_traj_size, bc_traj_size)
                        D_S = data_handler.traj_S[:, traj_idx].reshape([-1, data_handler.s_dim])
                        D_A = data_handler.traj_A[:, traj_idx].reshape([-1, data_handler.a_dim])
                        D_P_S_next = data_handler.traj_P_S_next[:, traj_idx].reshape([-1, data_handler.ps_dim])
                        D_masks = data_handler.traj_masks[:, traj_idx].reshape([-1, 1])
                    r_sas, r_sa, v_sas, v_sa = sess.run([galileo.r_sas_gen, galileo.r_sa_gen,
                                                            galileo.v_sas_op, galileo.v_sa_op],
                                                        feed_dict={galileo.gen_s_input_ph: D_S,
                                                                    galileo.gen_a_input_ph: D_A,
                                                                    galileo.gen_p_s_next_target_ph: D_P_S_next})
                    real_seg = {'R_SAS': r_sas.reshape([args.horizon, -1, 1]),
                                'R_SA': r_sa.reshape([args.horizon, -1, 1]),
                                'V_SAS': v_sas.reshape([args.horizon, -1, 2]),
                                'V_SA': v_sa.reshape([args.horizon, -1, 2]),
                                'V_next': v_sas.reshape([args.horizon, -1, 2])[-1] * 0,
                                'S': D_S.reshape([args.horizon, -1, data_handler.s_dim]),
                                'A': D_A.reshape([args.horizon, -1, data_handler.a_dim]),
                                'P_S_next': D_P_S_next.reshape([args.horizon, -1, data_handler.ps_dim]),
                                'D': ~D_masks.astype(np.bool).reshape([args.horizon, -1, 1])}
                    real_seg['D'][-1] = True
                    to_remove = ~D_masks.astype(np.bool).reshape([args.horizon, -1, 1])
                    add_vtarg_and_adv(real_seg, lam=galileo.lam, gamma=galileo.gamma, rollout_step=args.horizon,
                                        traj_size=bc_traj_size, adv_type='sas')
                    add_vtarg_and_adv(real_seg, lam=galileo.lam, gamma=galileo.gamma, rollout_step=args.horizon,
                                        traj_size=bc_traj_size, adv_type='sa')
                    # add_vtarg_and_adv(real_seg, lam=galileo.lam, gamma=galileo.gamma, rollout_step=args.horizon,
                    #                   traj_size=bc_traj_size, adv_type='mixed')
                    # TODO: 重新测试新版
                    real_seg['adv_mixed'] = real_seg['tdlamret_sas'] - real_seg['tdlamret_sa']
                    remove_masked_data(real_seg, to_remove)
                    D_ADV = real_seg['adv_mixed'].reshape([-1, 2])
                    recent_adv_max.append(D_ADV.max(axis=0))
                    recent_adv_min.append(D_ADV.min(axis=0))
                    if args.zero_mean:
                        D_ADV = (D_ADV - D_ADV.mean(axis=0)) / (np.mean(recent_adv_max, axis=0) - np.mean(recent_adv_min, axis=0))
                    else:
                        D_ADV = D_ADV / (np.mean(recent_adv_max, axis=0) - np.mean(recent_adv_min, axis=0))
                    # TODO: 重新测试新版（配套的adv - mean）
                    logger.record_tabular(f'weighted-bc/hist-atarg_max', np.mean(recent_adv_max))
                    logger.record_tabular(f'weighted-bc/hist-atarg_min', np.mean(recent_adv_min))
                    logger.record_tabular(f'weighted-bc/atarg', np.mean(D_ADV))
                    logger.record_tabular(f'weighted-bc/atarg_max', np.max(D_ADV))
                    logger.record_tabular(f'weighted-bc/atarg_min', np.min(D_ADV))
                
                    # if len(recent_adv_max) >= recent_adv_max.maxlen:
                    mse_idx = np.random.randint(0, real_seg['S'].shape[0], args.bc_batch_size)
                    v_model_likelihood, model_likelihood, v_sl_global_norm_out, \
                    pi_likelihood, pi_std, dm_std = sess.run([galileo.v_weight_model_likelihood_loss,
                                                                galileo.model_lilelihood_loss,
                                                                galileo.sl_global_norm, galileo.pi_likelihood_loss,
                                                                galileo.pi_std, galileo.dm_std,
                                                                galileo.v_weight_likelihood_op, galileo.pi_likelihood_op],
                                                                feed_dict={galileo.s_input_ph: real_seg['S'][mse_idx],
                                                                        galileo.a_input_ph: real_seg['A'][mse_idx],
                                                                        galileo.p_s_next_target_ph: real_seg['P_S_next'][mse_idx],
                                                                        galileo.atarg_ph: D_ADV[mse_idx],
                                                                        galileo.g_norm_ph: g_norm})[:-2]
                    logger.record_tabular("loss/v_model_neg_log_likelihood", np.mean(v_model_likelihood))
                    logger.record_tabular("grad/v_sl_global_norm_out", np.mean(v_sl_global_norm_out))
                        # else:
                        #     do_v_weight_mse = False
                    # else:
                else:
                    mse_idx = np.random.randint(0, data_handler.data_size, args.bc_batch_size)
                    model_likelihood, sl_global_norm_out, pi_likelihood,\
                    pi_std, dm_std = sess.run([galileo.model_lilelihood_loss,
                                                galileo.sl_global_no_v_norm, galileo.pi_likelihood_loss,
                                                galileo.pi_std, galileo.dm_std,
                                                galileo.model_likelihood_op, galileo.pi_likelihood_op],
                                                feed_dict={galileo.s_input_ph: data_handler.S[mse_idx],
                                                            galileo.a_input_ph: data_handler.A[mse_idx],
                                                            galileo.p_s_next_target_ph: data_handler.P_S_next[mse_idx],
                                                            galileo.g_norm_ph: g_norm})[:-2]
                    logger.record_tabular("grad/sl_global_norm_out", np.mean(sl_global_norm_out))
            if args.bc_step == 0:
                mse_idx = np.random.randint(0, data_handler.data_size, args.bc_batch_size)
                model_likelihood, pi_likelihood, pi_std, dm_std = sess.run([galileo.model_lilelihood_loss, galileo.pi_likelihood_loss,
                                           galileo.pi_std, galileo.dm_std],
                                          feed_dict={galileo.s_input_ph: data_handler.S[mse_idx],
                                                     galileo.a_input_ph: data_handler.A[mse_idx],
                                                     galileo.p_s_next_target_ph: data_handler.P_S_next[mse_idx]})
            exp_manager.time_record_end('BC')
            # TODO: discriminator update should be in the end of the iteration.
            exp_manager.time_record('D')
            for _ in range(args.d_step):
                dis_batch_size = args.dis_batch_traj_size * args.horizon

                # dis_batch_size = np.minimum(dis_batch_size, data_handler.data_size)
                # traj_shuffle_indexes = np.random.permutation(data_handler.data_traj_size)
                for idx_batch in range(int(np.minimum(np.ceil(data_handler.data_size / dis_batch_size), args.dis_mini_batch_num))):
                    dis_idx = np.random.randint(0, data_handler.data_size, dis_batch_size)
                    # traj_indexes = traj_shuffle_indexes[
                    #                idx_batch * args.dis_batch_traj_size:(idx_batch + 1) * args.dis_batch_traj_size]
                    # temp_D_S = np.asarray(data_handler.traj_S)[:, traj_indexes].reshape([-1, data_handler.s_dim])
                    # temp_D_A = np.asarray(data_handler.traj_A)[:, traj_indexes].reshape([-1, data_handler.a_dim])
                    # temp_D_P_S_next = np.asarray(data_handler.traj_P_S_next)[:, traj_indexes].reshape(
                    #     [-1, data_handler.ps_dim])

                    dis_loss_out_sas, dis_loss_out_sa, prob_fake_out_sas, prob_fake_out_sa, \
                    prob_real_out_sas, prob_real_out_sa, noise_prob_fake_out_sas, noise_prob_fake_out_sa, \
                    noise_prob_real_out_sas, noise_prob_real_out_sa = sess.run([galileo.dis_loss_sas,
                                                                            galileo.dis_loss_sa,
                                                                            galileo.prob_fake_sas, galileo.prob_fake_sa,
                                                                            galileo.prob_real_sas, galileo.prob_real_sa,
                                                                            galileo.prob_fake_sas_noise, galileo.prob_fake_sa_noise,
                                                                            galileo.prob_real_sas_noise, galileo.prob_real_sa_noise,
                                                                            galileo.dis_sas_opt_op,
                                                                            galileo.dis_sa_opt_op],
                                                                           feed_dict={galileo.s_input_ph: data_handler.S[dis_idx],
                                                                                      galileo.a_input_ph: data_handler.A[dis_idx],
                                                                                      galileo.p_s_next_target_ph: data_handler.P_S_next[dis_idx],
                                                                                      galileo.gen_s_input_ph: seg['S'],
                                                                                      galileo.gen_a_input_ph: seg['A'],
                                                                                      galileo.gen_p_s_next_target_ph: seg['P_S_next'],
                                                                                      })[:-2]

            if args.d_step > 0:
                logger.record_tabular("loss/dis_loss_out_sas", np.mean(dis_loss_out_sas), freq=20)
                logger.record_tabular("loss/dis_loss_out_sa", np.mean(dis_loss_out_sa), freq=20)
                logger.record_tabular("prob/fake_out_sas", np.mean(prob_fake_out_sas), freq=20)
                logger.record_tabular("prob/real_out_sas", np.mean(prob_real_out_sas), freq=20)
                logger.record_tabular("prob/fake_out_sa", np.mean(prob_fake_out_sa), freq=20)
                logger.record_tabular("prob/real_out_sa", np.mean(prob_real_out_sa), freq=20)
                logger.record_tabular("prob_noise/fake_out_sas", np.mean(noise_prob_fake_out_sas), freq=20)
                logger.record_tabular("prob_noise/real_out_sas", np.mean(noise_prob_real_out_sas), freq=20)
                logger.record_tabular("prob_noise/fake_out_sa", np.mean(noise_prob_fake_out_sa), freq=20)
                logger.record_tabular("prob_noise/real_out_sa", np.mean(noise_prob_real_out_sa), freq=20)

            exp_manager.time_record_end('D')

            logger.record_tabular(f'std/pi_std_mean', np.mean(pi_std), freq=20)
            logger.record_tabular(f'std/pi_std_max', np.max(pi_std), freq=20)
            logger.record_tabular(f'std/pi_std_min', np.min(pi_std), freq=20)
            logger.record_tabular(f'std/dm_std_mean', np.mean(dm_std), freq=20)
            logger.record_tabular(f'std/dm_std_max', np.max(dm_std), freq=20)
            logger.record_tabular(f'std/dm_std_min', np.min(dm_std), freq=20)
            logger.record_tabular(f'perf/do_v_weight_mse', do_v_weight_mse)

            logger.record_tabular("loss/model_neg_log_likelihood", np.mean(model_likelihood))
            # logger.record_tabular("loss/model_neg_log_likelihood_max", np.max(model_likelihood))
            # logger.record_tabular("loss/model_neg_log_likelihood_min", np.min(model_likelihood))
            logger.record_tabular("loss/pi_neg_log_likelihood", np.mean(pi_likelihood))

            if i % 2000 == 0:
                if real_values is None:
                    real_values = []
                    for idx, policy in enumerate(eval_policy_set):
                        logger.info(f"eval_idx {idx}")
                        real_value = compute_real_value(env.env, policy, num_eval_episodes=args.num_eval_episodes)
                        real_values.append(real_value)


                fake_values = []
                for idx, policy in enumerate(eval_policy_set):
                    logger.info(f"eval_idx {idx}")
                    fake_value = compute_real_value(dm_env_evaluation, policy, num_eval_episodes=args.num_eval_episodes)
                    fake_values.append(fake_value)
                    logger.record_tabular(f"perf_ope_details/fake_value_{idx}", fake_value)
                    logger.record_tabular(f"perf_ope_details_2/fake_value_gap_{idx}", real_values[idx] - fake_value)

                real_values, fake_values = np.array(real_values), np.array(fake_values)
                value_min, value_max = real_values.min(), real_values.max()
                norm_real_values = (real_values - value_min) / (value_max - value_min)
                norm_fake_values = (fake_values - value_min) / (value_max - value_min)
                logger.info(norm_real_values)
                logger.info(norm_fake_values)
                absolute_error = (np.abs(norm_real_values - norm_fake_values)).mean()
                raw_absolute_error = (np.abs(real_values - fake_values)).mean()
                rank_correlation = np.corrcoef(norm_real_values, norm_fake_values)[0, 1]
                top_idxs = np.argsort(norm_fake_values)[-1:]
                regret = norm_real_values.max() - norm_real_values[top_idxs].max()
                logger.record_tabular("perf_ope/absolute_error", absolute_error)
                logger.record_tabular("perf_ope/raw_absolute_error", raw_absolute_error)
                logger.record_tabular("perf_ope/rank_correlation", rank_correlation)
                logger.record_tabular("perf_ope/regret", regret)
                logger.dump_tabular()
            if i % 400 == 0 and i > 0:
                # exp_manager.time_record('evaluation')
                data_handler.evaluation(do_plot=True, predict_fn=galileo.evaluate_next_state, dis_pred_fn=galileo.dis_data)
                # exp_manager.time_record_end('evaluation')
                exp_manager.save_checkpoint()
            logger.dump_tabular()
        else:
            raise NotImplementedError
        exp_manager.time_record_end('iter')

if __name__ == '__main__':
    main()
