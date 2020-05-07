class Agent:
    
    def __init__(self, environment, policy, optimizer):
        self.policy = policy
        self.env = environment
        self.optimizer = optimizer

    def run_episode(self):

        state = self.env.reset()

        for time in range(1000):

            action = self.policy.select_action(state)
            state, reward, done, _ = self.env.step(action)

            # Save reward
            self.policy.episode_reward.append(reward)
            if done:
                break

    def update_policy(self, discount_factor=0.99):

        self.policy.train()

        self.optimizer.zero_grad()
        self.policy.calculate_loss(discount_factor).backward()
        self.optimizer.step()
