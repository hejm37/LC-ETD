import numpy as np


class BidirectionalChain:
    def __init__(self, states_number: int = 2,
                 start_state_number: int = 1,
                 reward: int = 1, **kwargs):
        # features should be implemented in the child class

        # self._states_number = states_number
        self._start_state_number = start_state_number
        # self._terminal = None
        self._state = None
        self.LEFT_ACTION = 0
        self.RIGHT_ACTION = 1
        self.NEVER_TERMINATE = False
        self.num_states = states_number
        # self._window = None
        self.reward = [reward, 0]

    def reset(self):
        self._state = np.random.randint(0, self._start_state_number)
        return self._state

    def step(self, action):
        if action == self.LEFT_ACTION:
            next_state = self._state - 1
        elif action == self.RIGHT_ACTION:
            next_state = self._state + 1
        else:
            raise ValueError("action should be 0 or 1")

        reward = 0
        if next_state < 0:
            next_state = 0
            reward = self.reward[0]
        elif next_state >= self.num_states:
            next_state = self.num_states - 1
            reward = self.reward[1]

        self._state = next_state
        return self._state, reward, False, {}

    def render(self):
        pass


if __name__ == '__main__':
    env = BidirectionalChain()
    env.reset()
    for step in range(1, 1000):
        action = np.random.randint(0, 2)
        state, reward, done, info = env.step(action)
        print(f"step: {step}, state: {state}, action: {action}, " \
              f"reward: {reward}, done: {done}, info: {info}")
        if done:
            break
