import numpy as np

from Environments.BidirectionalChain import BidirectionalChain
from Tasks.BaseTask import BaseTask


class TwoStateSimpleHighVariance(BaseTask, BidirectionalChain):

    def __init__(self, **kwargs):
        BaseTask.__init__(self, **kwargs)
        BidirectionalChain.__init__(self)
        self._resource_root_path = kwargs.get('resource_root_path', 'Resources')

        self.N = 2
        self.STATE_ONE = 0
        self.STATE_TWO = 1
        self.feature_rep = self.load_feature_rep()
        self.num_features = self.feature_rep.shape[1]
        self.num_steps = kwargs.get('num_steps', 20000)
        self.GAMMA = 0.8
        self.LMBDA = kwargs.get('lmbda', 0.0)
        self.pi = 0.6
        self.mu = 0.4
        self.behavior_dist = self.load_behavior_dist()
        self.target_dist = self.load_target_dist()
        self.state_values = self.load_state_values()
        self.num_policies = TwoStateSimpleHighVariance.num_of_policies()
        self.ABTD_xi_zero = 1
        self.ABTD_xi_max = 2

    @staticmethod
    def num_of_policies():
        return 1

    def load_feature_rep(self):
        # num_states x num_features x num_runs
        return np.array([[1.], [2.]], dtype=np.float32)

    def create_feature_rep(self):
        raise NotImplementedError

    def get_state_feature_rep(self, s):
        return self.feature_rep[s, :]

    def load_behavior_dist(self):
        # num_states
        self.behavior_dist = np.array([self.mu, 1 - self.mu])
        return self.behavior_dist

    def load_target_dist(self):
        # num_states
        self.target_dist = np.array([self.pi, 1 - self.pi])
        return self.target_dist

    def load_state_values(self):
        self.state_values = np.array([2.04, 1.44])
        return self.state_values

    def select_behavior_action(self, s):
        if np.random.random() < self.mu:
            return self.LEFT_ACTION
        else:
            return self.RIGHT_ACTION

    def select_target_action(self, s, policy_id=0):
        if np.random.random() < self.pi:
            return self.LEFT_ACTION
        else:
            return self.RIGHT_ACTION

    def get_pi(self, s, a):
        if a == self.LEFT_ACTION:
            return self.pi
        else:
            return 1 - self.pi

    def get_mu(self, s, a):
        if a == self.LEFT_ACTION:
            return self.mu
        else:
            return 1 - self.mu
