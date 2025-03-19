from itertools import product

import matplotlib.pyplot as plt
import numpy as np
from base_controllers import UniformInputGenerator
from deepc_controller import *
from deepc_utils import *


class SimpleSystem:
    def __init__(self, initial_state: float) -> None:
        self._state = initial_state

    def set_state(self, state):
        self._state = state

    def apply(self, control_input):
        self._state = self.logistic_fun(self._state) + control_input
        return self._state

    @staticmethod
    def logistic_fun(x):
        return 10 / (1 + np.exp(-x / 10)) - 5

    def measurement_dim(self):
        return 1

    def input_dim(self):
        return 1


class CosineSystem:
    def __init__(self, initial_state: float) -> None:
        self._state = initial_state

    def set_state(self, state):
        self._state = state

    def apply(self, control_input):
        self._state[0] = (
            self._state[0] - 0.5 * np.sin(self._state[0]) + control_input[0]
        )

        self._state[1] = self._state[1] + control_input[1]
        return self._state.copy()

    def measurement_dim(self):
        return 2

    def input_dim(self):
        return 2


def generate_data(
    system: SimpleSystem,
    num_trajs=100,
    traj_len=10,
    initial_state_bounds=None,
    bounds=None,
    seed=42,
    input_cmd=None,
) -> TrajectoryDataSet:
    rng = np.random.default_rng(seed)
    data = []
    for i in range(num_trajs):
        initial_state = (
            rng.uniform(-10, 10)
            if initial_state_bounds is None
            else rng.uniform(*initial_state_bounds)
        )
        # print(initial_state)
        system.set_state(initial_state)
        state_traj = [initial_state.copy()]
        input_traj = []
        for j in range(traj_len):
            if input_cmd is None:
                random_input = (
                    rng.uniform(-5, 5) if bounds is None else rng.uniform(*bounds)
                )
            else:
                random_input = input_cmd(state_traj[-1])
            measured_state = system.apply(random_input)
            state_traj.append(measured_state)
            input_traj.append(random_input)

        state_traj = state_traj[:-1]

        data.append(
            TrajectoryData(
                np.array(state_traj).reshape(-1, system.measurement_dim()),
                np.array(input_traj).reshape(-1, system.input_dim()),
            )
        )

    return TrajectoryDataSet(data, "random_data")


def main_cosys():
    pass

    len_past = 2
    len_fut = 30

    dims = DeePCDims(len_past, len_fut, 2, 2)
    hanke_gen = HankelMatrixGenerator(len_past, len_fut)

    system = CosineSystem(np.array([0, 0]))
    noise_levels = [
        ("test_set", 2.0, 3),
        ("low", 0.1, 42),
        ("medium", 0.5, 42),
        ("high", 1.5, 42),
    ]

    test_set = None

    # input_cmd = lambda x: 0.2 * np.ones(2) * np.cos(x)
    input_cmd = lambda x: 0.25 * np.ones(2) * (5 - x[0])
    for noise_level in noise_levels:
        data = generate_data(
            system,
            100,
            len_past + len_fut,
            ([-2.75, -5], [2.75, 5]),
            ([-noise_level[1], -0.1], [noise_level[1], 0.1]),
            seed=noise_level[2],
            input_cmd=input_cmd if noise_level[0] == "test_set" else None,
        )

        fig = plt.figure()
        ax = fig.add_subplot()
        for traj in data.dataset:
            x, y = traj.state_trajectory[:, 0], traj.state_trajectory[:, 1]
            x_plus = x[1:]
            ax.plot(x[:-1], x_plus)
            ax.scatter(x[0], x_plus[0])

        ax.set_xlabel("x")
        ax.set_ylabel("y")
        fig.savefig(
            f"figures/prediction_validation/simple_sys/trajs_{noise_level[0]}.pdf"
        )

        fig.clear()
        ax.clear()

        # Dont do stuff if we just got the test set
        if noise_level[0] == "test_set":
            test_set = hanke_gen.generate_hankel_matrices(data)
            continue

        fig, ax = plt.subplots()

        selector = IsoMapEmbeddedDistances(data, dims, n_neighbors=4, n_components=30)
        # selector = LkSelector(1)

        color = plt.rcParams["axes.prop_cycle"].by_key()["color"][0]
        H_u, H_y = hanke_gen.generate_hankel_matrices(data)
        for use_affine in [True, False]:
            residuals = []
            for num_idcs in [10, 15, 20, 30, 50]:
                res = 0
                for u_val, y_val in zip(test_set[0].T, test_set[1].T):
                    idcs, _ = selector(
                        u_val.reshape(-1, 1), y_val.reshape(-1, 1), H_u, H_y, None
                    )
                    idcs = idcs[:num_idcs]
                    H_u_sel, H_y_sel = H_u[:, idcs], H_y[:, idcs]
                    Y_p, Y_f = (
                        H_y_sel[: dims.T_past * dims.p, :],
                        H_y_sel[dims.T_past * dims.p :, :],
                    )

                    pred = predict(
                        u_val,
                        y_val,
                        H_u_sel,
                        Y_p,
                        Y_f,
                        dims,
                        use_affine=use_affine,
                        return_jac=False,
                    )

                    curr_res = (
                        np.linalg.norm(
                            y_val[dims.T_past * dims.p :].reshape(-1) - pred.reshape(-1)
                        )
                        ** 2
                    )
                    res += curr_res

                residuals.append(res)

            ax.plot(
                [10, 15, 20, 30, 50], residuals, "--" if use_affine else "-", c=color
            )

        fig.suptitle(f"Noise level {noise_level[0]}")
        ax.set_xlabel("Number of Columns")
        ax.set_ylabel("Prediction Error")
        fig.savefig(f"figures/prediction_validation/simple_sys/error_{noise_level[0]}")
        # plt.show()


def predict(u_val, y_val, H_u_sel, Y_p, Y_f, dims, use_affine=False, return_jac=False):
    H = (
        np.vstack((H_u_sel, Y_p, np.ones((1, Y_p.shape[-1]))))
        if use_affine
        else np.vstack((H_u_sel, Y_p))
    )
    z = (
        np.append(np.append(u_val, y_val[: dims.T_past * dims.p]), [1])
        if use_affine
        else np.append(u_val, y_val[: dims.T_past * dims.p])
    )
    jac = Y_f @ np.linalg.pinv(H)
    pred = jac @ z
    return (pred, jac) if return_jac else pred


def main_rc():
    traj_lengths = [
        (1, 1),
        (2, 2),
        (2, 8),
        (2, 18),
        (2, 28),
        (2, 38),
        (2, 48),
        (2, 73),
        (2, 98),
        (2, 148),
    ]
    for len_past, len_fut in traj_lengths:
        system = SimpleSystem(0)
        num_traj = 1000
        traj_len = 200
        data = generate_data(system, num_traj, traj_len)

        p = 1
        m = 1
        n = 4
        T_past = len_past
        T_fut = len_fut
        Q = 100
        R = 0.1
        g_regularization_1 = 10
        g_regularization_pi = 1000
        slack_cost = 100000
        controller_costs = DeePCCost(
            Q, R, slack_cost, g_regularization_1, g_regularization_pi
        )

        A_constr_x = np.array([[1], [-1]])
        b_constr_x = np.array([10, 10])
        A_constr_u = np.array([[1], [-1]])
        b_constr_u = np.array([5, 5])
        deepc_constr = DeePCConstraints(A_constr_u, b_constr_u, A_constr_x, b_constr_x)

        deepc_dims = DeePCDims(T_past, T_fut, p, m)

        # deepc = DataDrivenPredictiveController.create_from_data(
        #     data, deepc_dims, 200, controller_costs, deepc_constr, 0, False
        # )
        # selector = LkSelector(deepc_dims, order=2)
        # selector = CosineDistances()
        selector = LkFractional(10)
        deepc = SelectDeePC(
            data,
            deepc_dims,
            controller_costs,
            deepc_constr,
            0,
            False,
            selector_callback=selector,
            save_relative_contrast=True,
        )

        system.set_state(0)
        state = np.array([0])
        for i in range(10):
            action = deepc.compute_action(state, 3)
            state = np.array([system.apply(action)])
            print(f"new state: {state}")


def plot_relative_contrast():
    plt.rcParams.update(
        {
            "text.usetex": True,
            "font.family": "serif",
            "font.sans-serif": "Times",
            "font.size": 10,
        }
    )
    norm_folders = ["l1", "l2", "cosine_distance"]
    label_names = [r"$L_1$", r"$L_2$", "cos dist"]
    folder = "data/rc"
    fig, ax = plt.subplots(figsize=(4, 1.6), tight_layout=True)
    for norm_folder, label in zip(norm_folders, label_names):
        curr_folder = os.path.join(folder, norm_folder)
        files = [
            file
            for file in os.listdir(curr_folder)
            if os.path.isfile(os.path.join(curr_folder, file))
        ]
        traj_len = []
        rel_contrast = []
        for file in files:
            fn_components = file.split("_")
            num = int(fn_components[2].split(".")[0])
            traj_len.append(num)
            rel_contrast.append(
                np.average(np.loadtxt(os.path.join(curr_folder, file), delimiter=","))
            )

        # plt.rcParams["mathtext.fontset"] = "cm"
        # plt.rcParams["font.family"] = "STIXGeneral"
        # plt.rcParams.update({"font.size": 18})
        # plt.rcParams.update(
        #     {"text.usetex": True, "font.family": "Computer Modern Roman"}
        # )
        ax.scatter(traj_len, rel_contrast, label=label)

    ax.set_yscale("log")
    ax.set_xlabel("Trajectory Dimensionality")
    ax.set_ylabel(r"Relative Contrast $\Delta$")
    # ax.set_ylabel(
    #     r"Relative Contrast $\frac{d_{\textrm{max}} - d_{\textrm{min}}}{d_{\textrm{min}}}$"
    # )
    # - d_{\text{min}}}{d_{\text{min}}
    ax.legend()
    fig.tight_layout()
    fig.savefig("figures/relative_contrast.pdf")
    plt.show()


if __name__ == "__main__":
    # main()
    plot_relative_contrast()

    # main_cosys()
