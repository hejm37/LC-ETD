import numpy as np
import random

from Environments.FourRoomGridWorld import FourRoomGridWorld
from Tasks.BaseTask import BaseTask
from utils import ImmutableDict


class LearnFourPoliciesTileCodingFeatCont(BaseTask, FourRoomGridWorld):
    '''Continuing four room task'''

    def __init__(self, **kwargs):
        BaseTask.__init__(self)
        FourRoomGridWorld.__init__(self)
        self.feature_rep = self.load_feature_rep()
        self.num_features = self.feature_rep.shape[1]
        self.num_steps = kwargs.get('num_steps', 50000)
        self.GAMMA = 0.9
        self.LMBDA = kwargs.get('lmbda', 0.0)
        self.target_dist = self.load_target_dist()  
        self.state_values = self.load_state_values()
        self.ABTD_xi_zero = 1
        self.ABTD_xi_max = 4

        self.optimal_sub_policies = ImmutableDict(
            {
                0: [
                    [lambda x, y: 0 <= x <= 3 and 2 <= y <= 4, [self.ACTION_DOWN, self.ACTION_RIGHT]],
                    [lambda x, y: 3 >= x >= 0 == y, [self.ACTION_UP, self.ACTION_RIGHT]],
                    [lambda x, y: 0 <= x <= 4 and y == 1, [self.ACTION_RIGHT]],
                    [lambda x, y: x == self.hallways[1][0] and y == self.hallways[1][1], [self.ACTION_DOWN]],
                    [lambda x, y: 4 == x and 2 <= y <= 4, [self.ACTION_DOWN]],
                    [lambda x, y: 4 == x and y == 0, [self.ACTION_UP]]
                ],
                1: [
                    [lambda x, y: 2 <= x <= 4 and 0 <= y <= 3, [self.ACTION_LEFT, self.ACTION_UP]],
                    [lambda x, y: x == 0 and 0 <= y <= 3, [self.ACTION_RIGHT, self.ACTION_UP]],
                    [lambda x, y: x == 1 and 0 <= y <= 4, [self.ACTION_UP]],
                    [lambda x, y: x == self.hallways[0][0] and y == self.hallways[0][1], [self.ACTION_LEFT]],
                    [lambda x, y: 2 <= x <= 4 and y == 4, [self.ACTION_LEFT]],
                    [lambda x, y: x == 0 and y == 4, [self.ACTION_RIGHT]],
                ],
                2: [
                    [lambda x, y: 2 <= x <= 4 and 7 <= y <= 10, [self.ACTION_LEFT, self.ACTION_DOWN]],
                    [lambda x, y: x == 0 and 7 <= y <= 10, [self.ACTION_RIGHT, self.ACTION_DOWN]],
                    [lambda x, y: x == 1 and 6 <= y <= 10, [self.ACTION_DOWN]],
                    [lambda x, y: x == self.hallways[2][0] and y == self.hallways[2][1], [self.ACTION_LEFT]],
                    [lambda x, y: 2 <= x <= 4 and y == 6, [self.ACTION_LEFT]],
                    [lambda x, y: x == 0 and y == 6, [self.ACTION_RIGHT]],
                ],
                3: [
                    [lambda x, y: 0 <= x <= 3 and 6 <= y <= 7, [self.ACTION_UP, self.ACTION_RIGHT]],
                    [lambda x, y: 0 <= x <= 3 and 9 <= y <= 10, [self.ACTION_DOWN, self.ACTION_RIGHT]],
                    [lambda x, y: 0 <= x <= 4 and y == 8, [self.ACTION_RIGHT]],
                    [lambda x, y: x == self.hallways[1][0] and y == self.hallways[1][1], [self.ACTION_UP]],
                    [lambda x, y: x == 4 and 6 <= y <= 7, [self.ACTION_UP]],
                    [lambda x, y: x == 4 and 9 <= y <= 10, [self.ACTION_DOWN]]
                ],
                4: [
                    [lambda x, y: 10 >= x >= 7 >= y >= 5, [self.ACTION_LEFT, self.ACTION_UP]],
                    [lambda x, y: 7 <= x <= 10 and 9 <= y <= 10, [self.ACTION_LEFT, self.ACTION_DOWN]],
                    [lambda x, y: 6 <= x <= 10 and y == 8, [self.ACTION_LEFT]],
                    [lambda x, y: x == self.hallways[3][0] and y == self.hallways[3][1], [self.ACTION_UP]],
                    [lambda x, y: x == 6 and 5 <= y <= 7, [self.ACTION_UP]],
                    [lambda x, y: x == 6 and 9 <= y <= 10, [self.ACTION_DOWN]]
                ],
                5: [
                    [lambda x, y: 6 <= x <= 7 and 6 <= y <= 10, [self.ACTION_RIGHT, self.ACTION_DOWN]],
                    [lambda x, y: 9 <= x <= 10 and 6 <= y <= 10, [self.ACTION_DOWN, self.ACTION_LEFT]],
                    [lambda x, y: x == 8 and 5 <= y <= 10, [self.ACTION_DOWN]],
                    [lambda x, y: x == self.hallways[2][0] and y == self.hallways[2][1], [self.ACTION_RIGHT]],
                    [lambda x, y: 6 <= x <= 7 and y == 5, [self.ACTION_RIGHT]],
                    [lambda x, y: 9 <= x <= 10 and y == 5, [self.ACTION_LEFT]]
                ],
                6: [
                    [lambda x, y: 6 <= x <= 7 and 0 <= y <= 2, [self.ACTION_UP, self.ACTION_RIGHT]],
                    [lambda x, y: 9 <= x <= 10 and 0 <= y <= 2, [self.ACTION_UP, self.ACTION_LEFT]],
                    [lambda x, y: x == 8 and 0 <= y <= 3, [self.ACTION_UP]],
                    [lambda x, y: x == self.hallways[0][0] and y == self.hallways[0][1], [self.ACTION_RIGHT]],
                    [lambda x, y: 6 <= x <= 7 and y == 3, [self.ACTION_RIGHT]],
                    [lambda x, y: 9 <= x <= 10 and y == 3, [self.ACTION_LEFT]]
                ],
                7: [
                    [lambda x, y: 7 <= x <= 10 and 2 <= y <= 3, [self.ACTION_DOWN, self.ACTION_LEFT]],
                    [lambda x, y: 7 <= x <= 10 and y == 0, [self.ACTION_UP, self.ACTION_LEFT]],
                    [lambda x, y: 6 <= x <= 10 and y == 1, [self.ACTION_LEFT]],
                    [lambda x, y: x == self.hallways[3][0] and y == self.hallways[3][1], [self.ACTION_DOWN]],
                    [lambda x, y: x == 6 and 2 <= y <= 3, [self.ACTION_DOWN]],
                    [lambda x, y: x == 6 and y == 0, [self.ACTION_UP]]
                ]
            }
        )
        self.optimal_room_policies = {
            0: [
                [lambda x, y: 0 <= x <= 4 and 0 <= y <= 4, self.optimal_sub_policies[0]],
                [lambda x, y: 0 <= x <= 4 and 6 <= y <= 10, self.optimal_sub_policies[2]],
                [lambda x, y: 6 <= x <= 10 and 0 <= y <= 3, self.optimal_sub_policies[7]],
                [lambda x, y: 6 <= x <= 10 and 5 <= y <= 10, self.optimal_sub_policies[5]],
            ],
            1: [
                [lambda x, y: 0 <= x <= 4 and 0 <= y <= 4, self.optimal_sub_policies[1]],
                [lambda x, y: 0 <= x <= 4 and 6 <= y <= 10, self.optimal_sub_policies[2]],
                [lambda x, y: 6 <= x <= 10 and 0 <= y <= 3, self.optimal_sub_policies[7]],
                [lambda x, y: 6 <= x <= 10 and 5 <= y <= 10, self.optimal_sub_policies[4]],
            ],
            2: [
                [lambda x, y: 0 <= x <= 4 and 0 <= y <= 4, self.optimal_sub_policies[1]],
                [lambda x, y: 0 <= x <= 4 and 6 <= y <= 10, self.optimal_sub_policies[3]],
                [lambda x, y: 6 <= x <= 10 and 0 <= y <= 3, self.optimal_sub_policies[6]],
                [lambda x, y: 6 <= x <= 10 and 5 <= y <= 10, self.optimal_sub_policies[4]],
            ],
            3: [
                [lambda x, y: 0 <= x <= 4 and 0 <= y <= 4, self.optimal_sub_policies[0]],
                [lambda x, y: 0 <= x <= 4 and 6 <= y <= 10, self.optimal_sub_policies[3]],
                [lambda x, y: 6 <= x <= 10 and 0 <= y <= 3, self.optimal_sub_policies[6]],
                [lambda x, y: 6 <= x <= 10 and 5 <= y <= 10, self.optimal_sub_policies[5]],
            ]
        }
        self.optimal_hallway_policies = {
            0: [
                [lambda x, y: x == self.hallways[0][0] and y == self.hallways[0][1], [self.ACTION_DOWN, self.ACTION_UP]],
                [lambda x, y: x == self.hallways[1][0] and y == self.hallways[1][1], [self.ACTION_DOWN]],
                [lambda x, y: x == self.hallways[2][0] and y == self.hallways[2][1], [self.ACTION_LEFT, self.ACTION_RIGHT]],
                [lambda x, y: x == self.hallways[3][0] and y == self.hallways[3][1], [self.ACTION_DOWN]]
            ],
            1: [
                [lambda x, y: x == self.hallways[0][0] and y == self.hallways[0][1], [self.ACTION_LEFT]],
                [lambda x, y: x == self.hallways[1][0] and y == self.hallways[1][1], [self.ACTION_LEFT, self.ACTION_RIGHT]],
                [lambda x, y: x == self.hallways[2][0] and y == self.hallways[2][1], [self.ACTION_LEFT]],
                [lambda x, y: x == self.hallways[3][0] and y == self.hallways[3][1], [self.ACTION_DOWN, self.ACTION_UP]]
            ],
            2: [
                [lambda x, y: x == self.hallways[0][0] and y == self.hallways[0][1], [self.ACTION_LEFT, self.ACTION_RIGHT]],
                [lambda x, y: x == self.hallways[1][0] and y == self.hallways[1][1], [self.ACTION_UP]],
                [lambda x, y: x == self.hallways[2][0] and y == self.hallways[2][1], [self.ACTION_DOWN, self.ACTION_UP]],
                [lambda x, y: x == self.hallways[3][0] and y == self.hallways[3][1], [self.ACTION_UP]]
            ],
            3: [
                [lambda x, y: x == self.hallways[0][0] and y == self.hallways[0][1], [self.ACTION_RIGHT]],
                [lambda x, y: x == self.hallways[1][0] and y == self.hallways[1][1], [self.ACTION_DOWN, self.ACTION_UP]],
                [lambda x, y: x == self.hallways[2][0] and y == self.hallways[2][1], [self.ACTION_RIGHT]],
                [lambda x, y: x == self.hallways[3][0] and y == self.hallways[3][1], [self.ACTION_LEFT, self.ACTION_RIGHT]]
            ]
        }
        self.policy_terminal_condition = ImmutableDict(
            {
                0: lambda x, y: x == self.hallways[0][0] and y == self.hallways[0][1],
                1: lambda x, y: x == self.hallways[1][0] and y == self.hallways[1][1],
                2: lambda x, y: x == self.hallways[2][0] and y == self.hallways[2][1],
                3: lambda x, y: x == self.hallways[3][0] and y == self.hallways[3][1],
            }
        )
        self.num_policies = LearnFourPoliciesTileCodingFeatCont.num_of_policies()
        self.stacked_feature_rep = self.stack_feature_rep()
        self._stochasticity_fraction = 0.5
        self.epsilon = 0.2

    @staticmethod
    def num_of_policies():
        return 4

    @staticmethod
    def num_of_states():
        return 121

    def get_terminal_policies(self, s):
        '''HACK: Policies won't terminate but would receive a reward.'''
        # FIXME: Change the name to reflect the fact that it won't terminate.
        # See also self.get_active_policies()
        x, y = self.get_xy(s)
        terminal_policies = np.zeros(self.num_policies)
        for policy_id, condition in self.policy_terminal_condition.items():
            if condition(x, y):
                terminal_policies[policy_id] = 1
        return terminal_policies

    def get_state_index(self, x, y):
        return int(y * np.sqrt(self.feature_rep.shape[0]) + x)

    def get_optimal_policy(self, policy_id, x, y):
        optimal_policies = self.optimal_hallway_policies[policy_id]
        # If the current state is in a room, then use the room policy.
        for condition, sub_policies in self.optimal_room_policies[policy_id]:
            if condition(x, y):
                optimal_policies = sub_policies
                break
        return optimal_policies

    def get_possible_actions(self, policy_id, s):
        x, y = self.get_xy(s)
        optimal_policies = self.get_optimal_policy(policy_id, x, y)

        for condition, actions in optimal_policies:
            if condition(x, y):
                possible_actions = actions
                break
        return possible_actions

    def get_probability(self, policy_id, s, a):
        possible_actions = self.get_possible_actions(policy_id, s)
        # 10 percent prob of selecting a random action, each with 2.5 percent prob
        probability = self.epsilon / self.num_actions
        if a in possible_actions:
            probability += (1 - self.epsilon) / len(possible_actions)
        return probability

    def select_target_action(self, s, policy_id=0):
        if self.epsilon > np.random.rand():
            action = np.random.randint(0, self.num_actions)
        else:
            possible_actions = self.get_possible_actions(policy_id, s)
            action = np.random.choice(possible_actions)
        return action

    def get_active_policies(self, s):
        '''All policies are active at any state in this task.'''
        return np.ones(self.num_policies, dtype=int)

    def load_feature_rep(self):
        return np.load(f'Resources/{self.__class__.__name__}/feature_rep.npy')

    def get_state_feature_rep(self, s):
        return self.feature_rep[s, :]

    def create_feature_rep(self):
        ...

    def load_behavior_dist(self):
        '''Warning: This is inacurate. Do not use.'''
        ...

    def load_target_dist(self):
        return np.load(f'Resources/{self.__class__.__name__}/d_pi.npy')

    def load_state_values(self):
        return np.load(f'Resources/{self.__class__.__name__}/state_values.npy')

    def select_behavior_action(self, s):
        return np.random.randint(0, self.num_actions)

    def get_mu(self, s, a):
        return np.ones(self.num_policies) * (1.0 / self.num_actions)

    def get_pi(self, s, a):
        pi_vec = np.zeros(self.num_policies)
        for policy_id in range(self.num_policies):
            pi_vec[policy_id] = self.get_probability(policy_id, s, a)
        return pi_vec

    def generate_behavior_dist(self, total_steps):
        final_state_dist = np.zeros((self.num_policies, self.num_states))
        s = self.reset()
        state_visitation_count = np.zeros(self.num_states)
        for step in range(total_steps):
            if step % 100000 == 0:
                print(step)
            state_visitation_count[s] += 1
            sp, r, is_terminal, _ = self.step(self.select_behavior_action(s))
            s = sp
        for s in range(self.num_states):
            for policy_id, i in enumerate(self.get_active_policies(s)):
                if i:
                    final_state_dist[policy_id, s] = state_visitation_count[s]
        return (final_state_dist / total_steps).T

    def generate_target_dist(self, total_steps):
        final_state_dist = np.zeros((self.num_policies, self.num_states))
        for policy_id in range(self.num_policies):
            s = self.reset()
            state_visitation_count = np.zeros(self.num_states)
            for step in range(total_steps):
                if step % 100000 == 0:
                    print(step)
                state_visitation_count[s] += 1
                sp, r, is_terminal, _ = self.step(self.select_target_action(s, policy_id))
                s = sp
            final_state_dist[policy_id, :] = state_visitation_count / total_steps
            np.set_printoptions(precision=3)
            np.set_printoptions(suppress=True)
            print(final_state_dist[policy_id, :].reshape((11,11)))
        return final_state_dist.T
