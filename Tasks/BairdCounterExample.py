import itertools
from typing import Optional, Tuple, List

import numpy as np

from Environments.Baird import BairdsCounterExampleMDP
from Tasks.BaseTask import BaseTask


class BairdCounterExample(BaseTask, BairdsCounterExampleMDP):

    def __init__(self, **kwargs):
        BaseTask.__init__(self, **kwargs)
        BairdsCounterExampleMDP.__init__(self)
        self.feature_rep = self.load_feature_rep()
        self.num_features = self.feature_rep.shape[1]
        self.num_steps = kwargs.get('num_steps', 1000)
        self.GAMMA = 0.97
        self.pi = 1
        self.mu = 1 / 7
        self.theta_star = np.zeros(self.num_features)
        self.behavior_dist = self.load_behavior_dist()
        self.target_dist = self.load_target_dist()
        self.state_values = self.load_state_values()
        self.num_policies = BairdCounterExample.num_of_policies()

    @staticmethod
    def num_of_policies():
        return 1
    
    def load_feature_rep(self):
        return np.array(
            [
                [2, 0, 0, 0, 0, 0, 0, 1],
                [0, 2, 0, 0, 0, 0, 0, 1],
                [0, 0, 2, 0, 0, 0, 0, 1],
                [0, 0, 0, 2, 0, 0, 0, 1],
                [0, 0, 0, 0, 2, 0, 0, 1],
                [0, 0, 0, 0, 0, 2, 0, 1],
                [0, 0, 0, 0, 0, 0, 1, 2],
            ],
            dtype=np.float32
        )

    def get_state_feature_rep(self, s):
        return self.feature_rep[s, :]
    
    def load_behavior_dist(self):
        return np.array([1/7] * 7)

    def load_target_dist(self):
        # return np.array([0] * 6 + [1])
        return np.array([1/7] * 7)

    def load_state_values(self):
        return self.feature_rep.dot(self.theta_star)

    def select_behavior_action(self, s):
        if np.random.random() <= self.mu:
            return self._SOLID
        else:
            return self._DASHED

    def select_target_action(self, s, policy_id=0):
        if np.random.random() <= self.pi:
            return self._SOLID
        else:
            return self._DASHED

    def get_pi(self, s, a):
        if a == self._SOLID:
            return self.pi
        else:
            return 1 - self.pi

    def get_mu(self, s, a):
        if a == self._SOLID:
            return self.mu
        else:
            return 1 - self.mu
