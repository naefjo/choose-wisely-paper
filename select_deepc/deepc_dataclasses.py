from copy import deepcopy
from dataclasses import asdict, dataclass
from typing import Any, List

import numpy as np
from numpy.typing import NDArray


@dataclass
class TrajectoryData:
    state_trajectory: NDArray
    input_trajectory: NDArray

    def __post_init__(
        self,
    ):
        self.state_trajectory = np.atleast_2d(self.state_trajectory)
        self.input_trajectory = np.atleast_2d(self.input_trajectory)
        assert (
            np.atleast_2d(self.state_trajectory).shape[0]
            == np.atleast_2d(self.input_trajectory).shape[0]
        ), (
            f"encountered shapes obs: {self.state_trajectory.shape},"
            f" action: {self.input_trajectory.shape}. "
            "Expected (N, n_obs), (N, n_action)"
        )


@dataclass
class TrajectoryDataSet:
    dataset: List[TrajectoryData]
    dataset_name: str

    def __add__(self, other):
        new_dataset = deepcopy(self)
        new_dataset.dataset.extend(other.dataset)
        new_dataset.dataset_name += f"_plus_{other.dataset_name}"
        return new_dataset


@dataclass
class DeePCConstraints:
    A_u: NDArray
    b_u: NDArray
    A_y: NDArray
    b_y: NDArray


@dataclass
class DeePCCost:
    Q: NDArray
    R: NDArray
    slack_cost: float
    regularizer_cost_g_1: float
    regularizer_cost_g_pi: float

    def print(self):
        return (
            f"g_{int(self.regularizer_cost_g_1)}_pi_{int(self.regularizer_cost_g_pi)}"
        )

    def get_dict(self):
        n_q = self.Q.shape[0]
        n_r = self.R.shape[0]
        cost_dict = {f"q_{i}": self.Q[i, i] for i in range(n_q)}
        cost_dict.update({f"r_{i}": self.R[i, i] for i in range(n_r)})
        cost_dict.update(
            {
                "lambda_1": self.regularizer_cost_g_1,
                "lambda_pi": self.regularizer_cost_g_pi,
            }
        )
        return cost_dict


@dataclass
class DeePCDims:
    """Dimension struct for deepc

    p: measurement size
    m: input size
    """

    T_past: int
    T_fut: int
    p: int
    m: int

    def get_dict(self):
        return asdict(self)


@dataclass
class DeePCControllerArgs:
    trajectory_data: TrajectoryDataSet
    deepc_dims: DeePCDims
    T_hankel: int
    controller_costs: DeePCCost
    controller_constraints: DeePCConstraints
    input_reference: np.array
    verbose: bool

    def get_logging(self) -> dict[str, Any]:
        return {
            "dataset_name": self.trajectory_data.dataset_name,
            **self.deepc_dims.get_dict(),
            **self.controller_costs.get_dict(),
        }
