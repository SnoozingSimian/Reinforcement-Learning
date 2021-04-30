import numpy as np


class Bandit:
    def __init__(self, arms, bandit_type="stationary", mu=0.0, sigma=0.01,
                 q_star=None):
        # static variable, introduce variance in received rewards
        self.REWARD_VARIANCE = 1
        self.arms = arms
        self.mu = mu
        self.sigma = sigma

        # make sure only correct banit types are passed
        if (bandit_type == 'stationary' or
                bandit_type == 'non-stationary'):
            self.bandit_type = bandit_type
        else:
            raise NotImplementedError(f"`bandit_type` {bandit_type}" +
                                      " not implemented")

        # initialize q_star if provided, else set it to 0 for all arms
        if q_star is None:
            self.q_star = np.zeros(self.arms)
        else:
            if len(q_star) != self.arms:
                raise ValueError("length of `q_star` must be equal to" +
                                 " `arms`")
            else:
                self.q_star = q_star

    def __random_walk(self):
        self.q_star += np.random.normal(self.mu, self.sigma, self.arms)

    def pull_arm(self, arm):
        reward = np.random.normal(self.q_star[arm], self.REWARD_VARIANCE)
        if self.bandit_type == 'non-stationary':
            self.__random_walk()
        return reward

    def show_arms(self):
        return self.arms

    def get_best_arms(self):
        max_val = max(self.q_star)
        return [i for i, val in enumerate(self.q_star) if val == max_val]
