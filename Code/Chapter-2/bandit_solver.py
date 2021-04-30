import numpy as np

from bandit import Bandit


class BanditSolver:
    def __init__(self, bandit, max_iters):
        if isinstance(bandit, Bandit) is False:
            raise TypeError("parameter bandit should be an instance of " +
                            "the bandit class")
        self.bandit = bandit
        self.max_iters = max_iters

    def __argmax(self, x):
        max_val = max(x)
        indx = [i for i, val in enumerate(x) if val == max_val]
        return np.random.choice(indx)

    def __ucb_potential(self, Q, N, confidence, t):
        potential = np.zeros(len(Q))
        for i in range(len(Q)):
            if N[i] == 0:
                potential[i] = float('Inf')
            else:
                potential[i] = Q[i] + confidence*(np.sqrt(np.log(t) / N[i]))
        return potential

    def __eg_samp_avg(self, **kwargs):
        ARGS = ['epsilon']
        if ARGS != list(kwargs.keys()):
            raise ValueError("The epsilon greedy algorithm with sample " +
                             "average updates requires the following " +
                             f"arguments : \n{ARGS}")

        Q = np.zeros(self.bandit.show_arms())
        N = np.zeros(self.bandit.show_arms())
        rewards = np.zeros(self.max_iters)
        POA = np.zeros(self.max_iters)
        num_opt_actions = 0
        for i in range(self.max_iters):
            # select action subroutine
            coin_flip = np.random.binomial(1, kwargs['epsilon'])
            if coin_flip:
                A = np.random.randint(0, self.bandit.show_arms())
            else:
                A = self.__argmax(Q)

            N[A] += 1
            num_opt_actions += 1 if A in self.bandit.get_best_arms() else 0

            rewards[i] = self.bandit.pull_arm(A)
            POA[i] = round((num_opt_actions / (i + 1)), 2)
            Q[A] = (1/N[A])*(rewards[i] - Q[A])
        return rewards, POA

    def __eg_const_step(self, **kwargs):
        ARGS = ['epsilon', 'alpha']
        if ARGS != list(kwargs.keys()):
            raise ValueError("The epsilon greedy algorithm with constant " +
                             "step updates requires the following " +
                             f"arguments : \n{ARGS}")

        Q = np.zeros(self.bandit.show_arms())
        rewards = np.zeros(self.max_iters)
        POA = np.zeros(self.max_iters)
        num_opt_actions = 0
        for i in range(self.max_iters):
            # select action subroutine
            coin_flip = np.random.binomial(1, kwargs['epsilon'])
            if coin_flip:
                A = np.random.randint(0, self.bandit.show_arms())
            else:
                A = self.__argmax(Q)

            num_opt_actions += 1 if A in self.bandit.get_best_arms() else 0

            rewards[i] = self.bandit.pull_arm(A)
            POA[i] = round((num_opt_actions / (i + 1)), 2)
            Q[A] = (kwargs['alpha'])*(rewards[i] - Q[A])
        return rewards, POA

    def __g_opt(self, **kwargs):
        ARGS = ['Q_init', 'alpha']
        if ARGS != list(kwargs.keys()):
            raise ValueError("The greedy algorithm with optimistic " +
                             "initialization requires the following " +
                             f"arguments : \n{ARGS}")
        if len(kwargs['Q_init']) != self.bandit.show_arms():
            raise ValueError("Initial value estimates not of correct " +
                             "size")
        Q = kwargs['Q_init']
        rewards = np.zeros(self.max_iters)
        POA = np.zeros(self.max_iters)
        num_opt_actions = 0
        for i in range(self.max_iters):
            # select action subroutine
            A = self.__argmax(Q)
            num_opt_actions += 1 if A in self.bandit.get_best_arms() else 0
            rewards[i] = self.bandit.pull_arm(A)
            POA[i] = round((num_opt_actions / (i + 1)), 2)
            Q[A] = (kwargs['alpha'])*(rewards[i] - Q[A])
        return rewards, POA

    def __ucb(self, **kwargs):
        ARGS = ['confidence']
        if ARGS != list(kwargs.keys()):
            raise ValueError("The Upper Confidence Bound algorithm " +
                             "requires the following " +
                             f"arguments : \n{ARGS}")
        Q = np.zeros(self.bandit.show_arms())
        N = np.zeros(self.bandit.show_arms())
        rewards = np.zeros(self.max_iters)
        POA = np.zeros(self.max_iters)
        num_opt_actions = 0
        for i in range(self.max_iters):
            # select action subroutine
            potential = self.__ucb_potential(Q, N, kwargs['confidence'], i+1)
            A = self.__argmax(potential)
            N[A] += 1
            num_opt_actions += 1 if A in self.bandit.get_best_arms() else 0
            rewards[i] = self.bandit.pull_arm(A)
            POA[i] = round((num_opt_actions / (i + 1)), 2)
            Q[A] = (1/N[A])*(rewards[i] - Q[A])
        return rewards, POA

    def __grad(self, **kwargs):
        pass

    def solve(self, method, **kwargs):
        if method == 'EG_SAMP':
            return self.__eg_samp_avg(**kwargs)
        elif method == 'EG_CONST':
            return self.__eg_const_step(**kwargs)
        elif method == 'G_OPT':
            return self.__g_opt(**kwargs)
        elif method == 'UCB':
            return self.__ucb(**kwargs)
        elif method == 'GRAD':
            return self.__grad(**kwargs)
        else:
            raise NotImplementedError(f"method {method} is not available")
