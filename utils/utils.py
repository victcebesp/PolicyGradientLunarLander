import pandas as pd
import torch
from matplotlib import pyplot as plt


def discount_rewards(rewards, gamma):

    R = 0
    discounted_rewards = []
    for r in rewards[::-1]:
        R = r + gamma * R
        discounted_rewards.insert(0, R)
    return torch.FloatTensor(discounted_rewards)


def normalize_rewards(discounted_rewards):
    discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std())
    return discounted_rewards


def prepare_rewards(rewards, gamma):
    discounted_rewards = discount_rewards(rewards, gamma)
    normalized_rewards = normalize_rewards(discounted_rewards)
    return normalized_rewards


def plot_training_evolution(policy, episodes):
    window = int(episodes / 20)

    fig, ((ax1), (ax2)) = plt.subplots(2, 1, sharey=True, figsize=[9, 9])
    rolling_mean = pd.Series(policy.reward_history).rolling(window).mean()
    std = pd.Series(policy.reward_history).rolling(window).std()
    ax1.plot(rolling_mean)
    ax1.fill_between(range(len(policy.reward_history)), rolling_mean - std, rolling_mean + std, color='orange',
                     alpha=0.2)
    ax1.set_title('Episode Length Moving Average ({}-episode window)'.format(window))
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Episode Length')

    ax2.plot(policy.reward_history)
    ax2.set_title('Episode Length')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Episode Length')

    fig.tight_layout(pad=2)
    plt.show()