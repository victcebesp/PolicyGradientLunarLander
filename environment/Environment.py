import gym


class Environment:

    def __init__(self, environment_name):
        self.env = gym.make(environment_name)

    def reset(self):
        return self.env.reset()

    def step(self, action):
        action = action.unsqueeze(dim=0).numpy()[0]
        return self.env.step(action)

    def is_solved(self, mean_reward):
        return mean_reward >= self.env.spec.reward_threshold

    def observation_space(self):
        return self.env.observation_space.shape[0]

    def action_space(self):
        return self.env.action_space.n

    def render(self):
        self.env.render()

    def close(self):
        self.env.close()
