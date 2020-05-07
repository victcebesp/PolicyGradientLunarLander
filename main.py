import numpy as np
import torch
import torch.optim as optim

from agent.Agent import Agent
from environment.Environment import Environment
from policy.Policy import Policy
from utils.utils import plot_training_evolution

learning_rate = 0.01
discount_factor = 0.99
episodes = 5000

env = Environment('LunarLander-v2')
policy: Policy = Policy(env.observation_space(), env.action_space())
optimizer = optim.Adam(policy.parameters(), lr=learning_rate)
agent = Agent(env, policy, optimizer)

for episode in range(episodes):

    agent.run_episode()
    agent.update_policy(discount_factor=discount_factor)

    if episode % 50 == 0:
        print('Episode {}\tAverage reward: {}'.format(episode, np.array(policy.reward_history[-50:]).mean()))

        if env.is_solved(np.array(policy.reward_history[-50:]).mean()):
            break

torch.save(policy.state_dict(), 'saved_policy/policy.pt')

# Plot training process
plot_training_evolution(policy, episodes)
