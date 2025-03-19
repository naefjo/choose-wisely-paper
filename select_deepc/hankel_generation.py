import warnings
from typing import Optional, Tuple, Union

import numpy as np
from numpy.typing import NDArray

from .deepc_utils import DeePCDims, TrajectoryData, TrajectoryDataSet


class HankelMatrixGenerator:
    """Class which constructs the data model used in the DeePController"""

    def __init__(
        self, T_past: int = 2, T_fut: int = 10, T_hankel: Optional[int] = None
    ):
        """initialize Hankel matrix generator
        args:
            T_past: how many datapoints in to use to capture the past trajectory
            T_fut: how many datapoints into the future should be predicted
            T_hankel: The number of datapoints used to build the Hankel matrix.
              e.g. Hankel matrix will have T_hankel - T_past - T_fut columns.
        """
        self.L = T_past + T_fut
        self.T_hankel = T_hankel

    @staticmethod
    def block_hankel(w: np.array, L: int) -> np.array:
        """
        Builds block Hankel matrix of order L from data w
        args:
            w : a T x d data matrix. T is the number of timesteps, d is the dimension of the signal
              e.g., if there are 6 timesteps and 4 entries at each timestep, w is 6 x 4
            L : order of hankel matrix
        """
        T = int(np.shape(w)[0])  # number of timesteps
        d = int(np.shape(w)[1])  # dimension of the signal
        if L > T:
            raise ValueError(f"L {L} must be smaller than T {T}")

        H = np.zeros((L * d, T - L + 1))
        w_vec = w.reshape(-1)
        for i in range(0, T - L + 1):
            H[:, i] = w_vec[d * i : d * (L + i)]
        return H

    def generate_hankel_matrices(
        self,
        data: Union[TrajectoryDataSet, TrajectoryData],
        subsample_factor: float = 1.0,
    ) -> Tuple[NDArray, NDArray]:
        if isinstance(data, TrajectoryDataSet):
            H_u, H_y = None, None
            for trajectory in data.dataset:
                H_u_tmp, H_y_tmp = self.generate_hankel_matrices(trajectory)
                if H_u_tmp is None:
                    continue

                if H_u is None:
                    H_u = H_u_tmp
                    H_y = H_y_tmp
                else:
                    H_u = np.hstack((H_u, H_u_tmp))
                    H_y = np.hstack((H_y, H_y_tmp))

            idcs_to_keep = np.linspace(
                0,
                H_u.shape[-1],
                int(H_u.shape[-1] * subsample_factor),
                dtype=int,
                endpoint=False,
            )
            return H_u[:, idcs_to_keep], H_y[:, idcs_to_keep]

        assert isinstance(data, TrajectoryData), f"got instance {type(data)},"
        if data.input_trajectory.shape[0] < self.L:
            warnings.warn(
                f"trajectory too short ({data.input_trajectory.shape[0]}) for the desired horizon ({self.L})"
            )
            return None, None
        if (
            self.T_hankel is not None
            and data.input_trajectory.shape[0] <= self.T_hankel
        ):
            warnings.warn(
                f"input data sequence of lenght {data.input_trajectory.shape[0]} is "
                + "smaller than requested number of datapoints "
                + f"in the Hankel matrix {self.T_hankel}, creating shorter hankel matrix"
            )
        input_data = (
            data.input_trajectory[: self.T_hankel, :]
            if self.T_hankel is not None
            else data.input_trajectory
        )
        output_data = (
            data.state_trajectory[: self.T_hankel, :]
            if self.T_hankel is not None
            else data.state_trajectory
        )
        H_u = HankelMatrixGenerator.block_hankel(input_data, self.L)
        H_y = HankelMatrixGenerator.block_hankel(output_data, self.L)

        return H_u, H_y


def generate_multistep_predictor_alt(H_u, H_y, dims: DeePCDims):
    tp = dims.T_past
    tf = dims.T_fut
    p = dims.p  # y
    m = dims.m  # u

    Y_p, Y_f = np.vsplit(H_y, [tp * p])
    U_p, U_f = np.vsplit(H_u, [tp * m])

    Phi = np.zeros((tf * p, (tp + tf) * (m + p)))
    for i in range(tf):
        # (tp*m + tp*p + i*m)
        Z_lk = np.vstack((U_p, Y_p, U_f[: (i + 1) * m, :]))
        Y_rpk = Y_f[i * p : (i + 1) * p, :]
        Phi[i * p : (i + 1) * p, : tp * (m + p) + (i + 1) * m] = Y_rpk @ np.linalg.pinv(
            Z_lk
        )

    Phi_p = Phi[:, : tp * (m + p)]
    Phi_u = Phi[:, tp * (m + p) : tp * (m + p) + tf * m]
    Phi_y = Phi[:, tp * (m + p) + tf * m :]
    eye_min_phi_y_inv = np.linalg.inv(np.eye(Phi.shape[0]) - Phi_y)
    tilde_h = np.hstack((eye_min_phi_y_inv @ Phi_p, eye_min_phi_y_inv @ Phi_u))
    return tilde_h
