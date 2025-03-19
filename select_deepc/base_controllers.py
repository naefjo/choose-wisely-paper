from abc import ABC, abstractmethod
from typing import Tuple, Union

import numpy as np
from numpy.typing import NDArray


class BaseController(ABC):
    """Abstract base class which implements a generic controller interface"""

    @abstractmethod
    def compute_action(self, state: NDArray, reference: NDArray) -> NDArray:
        pass


class UniformInputGenerator(BaseController):
    """Samples input from a uniform distribution without stabilization"""

    def __init__(self, min_input: NDArray, max_input: NDArray, seed=None):
        self._rng = np.random.default_rng(seed=seed)
        self._min = min_input
        self._max = max_input

    def compute_action(self, state: NDArray, reference: NDArray) -> NDArray:
        return self._rng.uniform(self._min, self._max)


class RandomWalkInputGenerator(BaseController):
    """Samples input from a uniform distribution without stabilization"""

    def __init__(
        self,
        min_input: NDArray,
        max_input: NDArray,
        initial_input: NDArray,
        seed=None,
        innovation=0.1,
        forgetting_factor=1,
    ):
        self._rng = np.random.default_rng(seed=seed)
        self._prev_input = initial_input
        self._max = max_input
        self._min = min_input
        self._innovation = innovation
        self._forgetting_factor = forgetting_factor
        self._velocity = np.zeros_like(self._prev_input)

    def compute_action(self, state: NDArray, reference: NDArray) -> NDArray:
        new_delta_input = self._rng.normal(
            np.zeros_like(self._prev_input),
            self._innovation * np.ones_like(self._prev_input),
        )
        self._velocity = self._velocity + new_delta_input
        print(self._velocity)

        self._prev_input = np.clip(
            self._forgetting_factor * self._prev_input + self._velocity,
            self._min,
            self._max,
        )
        return self._prev_input
