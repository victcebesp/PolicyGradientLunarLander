import torch

from environment.Environment import Environment
from policy.Policy import Policy

env = Environment('LunarLander-v2')

policy: Policy = Policy(env.observation_space(), env.action_space())
policy.load_state_dict(torch.load('saved_policy/policy.pt'))
policy.eval()

for episode in range(500):
    state = env.reset()
    done = False

    for time in range(1000):
        action = policy.select_action(state)
        state, reward, done, _ = env.step(action)
        env.render()

        if done:
            break

    env.close()
