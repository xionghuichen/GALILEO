# Created by xionghuichen at 2022/7/26
# Email: chenxh@lamda.nju.edu.cn


def runner(S_init, traj_size, rollout_H, real_s, real_a, real_next_s):
    # np.random.shuffle(idx)

    S_train = []
    A_train = []
    R_SAS_train = []
    MSE_REW_train = []
    R_SA_train = []
    P_S_next_train = []
    V_SAS_train = []
    V_SA_train = []
    S_next_train = []
    sample_idx = np.random.randint(0, S_init.shape[0], traj_size)
    s = S_init[sample_idx]
    real_s = real_s[:, sample_idx]
    real_a = real_a[:, sample_idx]
    real_next_s = real_next_s[:, sample_idx]
    # s = np.repeat(s, traj_size, axis=0)
    D_train = []
    to_remove = []
    have_done = np.zeros((s.shape[0], 1))
    i = 0
    for i in range(int(rollout_H)):
        # TODO: test dis true_a
        # {
        if args.dual_train:
            a = sess.run(gen_a_sample, feed_dict={gen_s_input: s})
        else:
            a = policy_func(s)
        # a = real_a[i]
        # }

        # if args.policy_sample_noise > 0:
        #     a += np.random.normal(np.zeros(a.shape), np.ones(a.shape) * args.policy_sample_noise)
        # TODO: 可以测测转移随机性对结果的影响
        # TODO: test dis true_ans
        # {
        p_s_next = sess.run(gen_p_s_next_pred_sample, feed_dict={gen_s_input: s, gen_a_input: a})
        # p_s_next = part_env(s, a)
        # p_s_next = np.concatenate((part_env(s, a), comple_env(s, a, s)), axis=-1)
        # }
        if args.double_d:
            r_sas, r_sa, v_sas, v_sa = sess.run([rew_sas_op, rew_sa_op, v_sas_op, v_sa_op],
                                                feed_dict={gen_s_input: s, gen_a_input: a,
                                                           gen_p_s_next_target: p_s_next})
        else:
            r_sas, v_sas = sess.run([rew_sas_op, v_sas_op],
                                    feed_dict={gen_s_input: s, gen_a_input: a, gen_p_s_next_target: p_s_next})
            r_sa = r_sas
            v_sa = v_sas

        # s_next = comple_env(s, a, p_s_next)
        if not args.clip_switch:
            s_next = comple_env(s, a, p_s_next)  # TODO: Ablation
        else:
            s_next = comple_env(s, np.clip(a, a_min, a_max), np.clip(p_s_next, psmin, psmax))
        d = is_terminal(s, a, s_next, args.env_name)
        to_remove.append(have_done)
        have_done = np.clip(have_done + d, 0, 1)
        MSE_REW_train.append(mse_rew)
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
        if np.mean(have_done) == 1.0:
            break
    if not args.branch_init:
        D_train[-1] = np.ones(r_sas.shape)
    # if args.dual_train:
    #     a = sess.run(gen_a_sample, feed_dict={gen_s_input: s})
    # else:
    #     a = policy_func(s)
    seg = {
        "S": S_train,
        "A": A_train,
        "R_SAS": R_SAS_train,
        "R_SA": R_SA_train,
        "D": D_train,
        "V_SAS": V_SAS_train,
        "V_SA": V_SA_train,
        "P_S_next": P_S_next_train,
        "MSE_REW_train": MSE_REW_train,
        "V_next": np.zeros(v_sas.shape),  # the last step must be a done step
    }

    add_vtarg_and_adv(seg, args.gamma, args.lam, i + 1, traj_size, adv_type='sas')
    add_vtarg_and_adv(seg, args.gamma, args.lam, i + 1, traj_size, adv_type='sa')
    add_vtarg_and_adv(seg, args.gamma, args.lam, i + 1, traj_size, adv_type='mixed')
    to_remove = np.array(to_remove).astype(np.bool)[..., 0]
    for k in seg.keys():
        seg[k] = np.asarray(seg[k])
        if len(seg[k].shape) == 3:
            seg[k] = seg[k][np.where(~to_remove)]
        else:
            seg[k] = seg[k].reshape([-1, seg[k].shape[-1]])
    return seg