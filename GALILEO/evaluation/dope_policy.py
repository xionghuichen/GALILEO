import os
import pickle
import numpy as np
import tensorflow as tf
import torch


class D4RLPolicy:
    """D4RL policy."""

    def __init__(self, policy_file, device="cpu"):
        self.device = torch.device(device)
        with open(policy_file, 'rb') as f:
            weights = pickle.load(f)
        self.fc0_w = torch.from_numpy(weights['fc0/weight']).to(self.device)
        self.fc0_b = torch.from_numpy(weights['fc0/bias']).to(self.device)
        self.fc1_w = torch.from_numpy(weights['fc1/weight']).to(self.device)
        self.fc1_b = torch.from_numpy(weights['fc1/bias']).to(self.device)
        self.fclast_w = torch.from_numpy(weights['last_fc/weight']).to(self.device)
        self.fclast_b = torch.from_numpy(weights['last_fc/bias']).to(self.device)
        self.fclast_w_logstd = torch.from_numpy(weights['last_fc_log_std/weight']).to(self.device)
        self.fclast_b_logstd = torch.from_numpy(weights['last_fc_log_std/bias']).to(self.device)
        # relu = lambda x: torch.maximum(x, 0)
        self.nonlinearity = torch.tanh if weights['nonlinearity'] == 'tanh' else torch.relu

        identity = lambda x: x
        self.output_transformation = torch.tanh if weights[
            'output_distribution'] == 'tanh_gaussian' else identity

    def select_action(self, state, deterministic=False):
        # if torch.is_tensor(state): state = state.cpu().numpy()
        if len(state.shape) == 1:
            state = np.expand_dims(state, axis=0)
        state = torch.as_tensor(state, dtype=torch.float32).to(self.device)
        x = torch.mm(state, self.fc0_w.T) + self.fc0_b
        x = self.nonlinearity(x)
        x = torch.mm(x, self.fc1_w.T) + self.fc1_b
        x = self.nonlinearity(x)
        mean = torch.mm(x, self.fclast_w.T) + self.fclast_b
        logstd = torch.mm(x, self.fclast_w_logstd.T) + self.fclast_b_logstd
        if deterministic:
            action = self.output_transformation(mean)
        else:
            noise = torch.ones_like(logstd)
            action = self.output_transformation(mean + torch.exp(logstd) * noise)
        return action.cpu().numpy()


class DMCPolicy():
    def __init__(self, policy, obs_map_fn) -> None:
        self.policy = policy
        self.obs_map_fn = obs_map_fn

    def select_action(self, obs, deterministic):
        obs = self.obs_map_fn(obs)
        if hasattr(self.policy, 'initial_state'):
            action = self.policy(obs, ((),))[0]
        else:
            action = self.policy(obs)
        return action.numpy()


if __name__ == "__main__":
    obss = np.random.randn(256, 11)
    policy = D4RLPolicy(policy_file="hopper/hopper_online_0.pkl", device="cuda:2")
    actions = policy.select_action(obss)
    print(actions)