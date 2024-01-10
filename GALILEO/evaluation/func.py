# Created by xionghuichen at 2023/4/28
# Email: chenxh@lamda.nju.edu.cn
import numpy as np

def compute_real_value(real_env, policy, gamma=0.995, num_eval_episodes=10):
    # eval in real env
    real_ep_info = []
    obs = real_env.reset()
    num_episodes = 0
    value, episode_length = 0, 0
    repeat_times = 0
    while num_episodes < num_eval_episodes:
        action = policy.select_action(obs, deterministic=False)
        next_obs, reward, terminal, info = real_env.step(action.flatten())
        value += reward * (gamma ** episode_length)
        episode_length += 1
        obs = next_obs.copy()
        # if 'ood' in info and info['ood'] and episode_length < 200:
        #     print("reach ood at", episode_length, "reset times", repeat_times)
        #     print("info", info['ood_info'])
        #     if repeat_times >= 10:
        #         return -1000
        #     value, episode_length = 0, 0
        #     obs = real_env.reset()
        #     repeat_times += 1
        if terminal or episode_length >= 1000:
            real_ep_info.append(
                {"value": value, "episode_length": episode_length}
            )
            print(num_episodes, ": value", value, "episode_length",  episode_length)
            num_episodes += 1
            value, episode_length = 0, 0
            obs = real_env.reset()

    real_value = np.mean([info["value"] for info in real_ep_info])
    return real_value

def eval_value_gap(real_env, dynamics_model, policy, gamma=0.995, num_eval_episodes=10):
    # eval in fake env
    fake_ep_info = []
    obs = np.expand_dims(real_env.reset(), axis=0)

    num_episodes = 0
    value, episode_length = 0, 0
    h_state = None

    while num_episodes < num_eval_episodes:
        action = policy.select_action(obs, deterministic=False)
        next_obs, reward, terminal, info = dynamics_model.step(obs, action)
        reward, terminal = reward.flatten()[0], terminal.flatten()[0]
        value += reward * (gamma ** episode_length)
        episode_length += 1
        obs = next_obs.copy()
        if terminal or episode_length >= 1000:
            fake_ep_info.append(
                {"value": value, "episode_length": episode_length}
            )
            num_episodes += 1
            value, episode_length = 0, 0
            obs = np.expand_dims(real_env.reset(), axis=0)
    fake_value = np.mean([info["value"] for info in fake_ep_info])

    return fake_value