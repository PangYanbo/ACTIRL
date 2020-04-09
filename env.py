import numpy as np
import time


class Env(object):

    def __init__(self, n_step, n_states, n_actions, n_features):
        self.n_step = n_step
        self.states = [str(i) for i in range(n_states)]
        self.actions = [str(i) for i in range(n_actions)]
        self.n_states, self.n_actions, self.n_features = n_states, n_actions, n_features
        self.alpha = np.zeros(n_features)
        self.init = []

    def feature_vector(self, series):
        fv = np.zeros(self.n_features)
        # count of each activity, 0 ~ n_states-1
        for i in range(self.n_states):
            fv[i] = series.count(str(i))
         # count of activity types, n_states
        fv[self.n_states] = len(set(series))
         # check if last state is 'home'
        fv[self.n_states+1] = 1 if (len(series) == self.n_step and series[-1] == '0') else 0
         # series length
        fv[self.n_states+2] = len(series)
        return fv

    # def feature_vector(self, series):
    #     fv = np.zeros(self.n_states)
    #     for e in series:
    #         v = np.zeros(self.n_states)
    #         v[int(e)] = 1
    #         fv += v
    #     return fv

    def get_reward(self, series):
        return np.dot(self.feature_vector(series), self.alpha)

    def set_alpha(self, alpha):
        self.alpha = alpha

    def step(self, a):
        return a

    def get_init(self, demo):
        init = []
        for e in demo:
            init.append(e[0])
        self.init = init

    def reset(self):
        return np.random.choice(self.init)

