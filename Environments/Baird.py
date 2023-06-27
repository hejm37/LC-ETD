from typing import Tuple

import numpy as np


class BairdsCounterExampleMDP:
    """
        Adopted from
        https://gist.github.com/GiovanniGatti/ad20ba1c38d31211419bba7b17bebeb6
    """

    def __init__(self, **kwargs):
        self._DASHED = 0
        self._SOLID = 1
        self._ACTIONS = [self._DASHED, self._SOLID]
        self._STATES = np.arange(0, 7)
        self._LOWER_STATE = 6

    def reset(self):
        return np.random.choice(self._STATES)

    def step(self, action: int) -> Tuple[int, int]:
        """
        From the current state, move to the next state.

        Args:
            action (int): 0 dashed, 
                        1 solid 
        Returns:
            Tuple containing the index of the next state and the reward
        """
        ### START CODE HERE ### (≈ 2 lines of code)
        if action == self._DASHED:
            next_state = np.random.choice(self._STATES[: self._LOWER_STATE])
        else:
            next_state = self._LOWER_STATE
        return next_state, 0, False, {}
        ### END CODE HERE ###
    
    def num_actions(self) -> int:
        return len(self._ACTIONS)

    def num_states(self) -> int:
        return len(self._STATES)

    def render(self):
        pass


if __name__ == '__main__':
    env = BairdsCounterExampleMDP()
    env.reset()
    for step in range(1, 1000):
        action = np.random.randint(0, 2)
        state, reward, done, info = env.step(action)
        print(f"step: {step}, state: {state}, action: {action}, " \
              f"reward: {reward}, done: {done}, info: {info}")
        if done:
            break
