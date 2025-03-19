from abc import ABC

import numpy as np
from numpy.typing import NDArray


class SetPointScheduler(ABC):
    def __init__(self, env):
        pass

    def __call__(self, state, **kwargs):
        pass

    def is_successful(self, state):
        pass


class LandingScheduler(SetPointScheduler):
    def __init__(self, env):
        self._landing_position = env.get_landing_position()
        self._landing_position = np.append(self._landing_position, np.zeros(3))
        self._approach_hover_point = True
        self._landing_offset = np.zeros_like(self._landing_position)
        self._landing_offset[1] = 5

    def __call__(self, state, **kwargs) -> NDArray:
        x_pos_below_threshold = abs(state[0] - self._landing_position[0]) < 2
        vel_below_threshold = state[2] ** 2 + state[3] ** 2 < 1
        if x_pos_below_threshold and vel_below_threshold:
            self._approach_hover_point = False

        if self._approach_hover_point:
            return self._landing_position + self._landing_offset

        return self._landing_position

    def is_successful(self, state):
        has_ground_contact = np.any(state[6:])
        return has_ground_contact


class HoverSetPointScheduler(SetPointScheduler):
    def __init__(self, env):
        landing_position = env.get_landing_position()
        landing_position = np.append(landing_position, np.zeros(3))
        hover_offset = np.zeros_like(landing_position)
        hover_offset[1] = 5
        self._hover_position = landing_position + hover_offset

    def __call__(self, state, **kwargs) -> NDArray:
        return self._hover_position

    def is_successful(self, state):
        print(state)
        has_crashed = any(
            [
                abs(state[4]) > 0.5,
                state[0] < 5,
                state[0] > 25,
                state[1] > 25,
                state[1] < 5,
            ]
        )
        return not has_crashed
