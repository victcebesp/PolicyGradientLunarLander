import numpy as np
import torch
from torch import nn as nn
from torch.distributions import Categorical

from utils.utils import prepare_rewards


class Policy(nn.Module):

    def __call__(self, *input, **kwargs):
        return super().__call__(*input, **kwargs)

    def __init__(self, observation_space, action_space):
        super(Policy, self).__init__()
        self.state_space = observation_space
        self.action_space = action_space

        self.l1 = nn.Linear(self.state_space, 64, bias=False)
        self.l2 = nn.Linear(64, self.action_space, bias=False)

        self.episode_actions_probabilities = torch.Tensor()
        self.episode_reward = []
        self.reward_history = []

    def forward(self, state):
        model = torch.nn.Sequential(
            self.l1,
            nn.Dropout(p=0.6),
            nn.ReLU(),
            self.l2,
            nn.Softmax(dim=-1)
        )

        return model(torch.from_numpy(state))

    def select_action(self, state):

        self.eval()

        actions_probabilities = self(state)
        actions_distribution = Categorical(actions_probabilities)
        selected_action = actions_distribution.sample()

        if len(self.episode_actions_probabilities) > 0:
            self.episode_actions_probabilities = torch.cat([self.episode_actions_probabilities, actions_distribution.log_prob(selected_action).reshape(1)])
        else:
            self.episode_actions_probabilities = actions_distribution.log_prob(selected_action).reshape(1)

        return selected_action

    def calculate_loss(self, discount_factor):
        rewards = prepare_rewards(self.episode_reward, discount_factor)
        loss = torch.sum(torch.mul(self.episode_actions_probabilities, rewards)) * -1

        self.episode_actions_probabilities = torch.Tensor()
        self.reward_history.append(np.sum(self.episode_reward))
        self.episode_reward = []

        return loss
