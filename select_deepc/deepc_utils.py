import base64
import glob
import io
import os
from abc import ABC
from copy import deepcopy
from time import perf_counter
from typing import Optional

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from IPython import display as ipythondisplay
from IPython.display import HTML
from numpy.typing import NDArray

from .base_controllers import BaseController
from .deepc_dataclasses import *
from .deepc_rocket_utils import SetPointScheduler


class PerformanceAccumulator(ABC):
    def update_cost(
        self, state: np.ndarray, reference: np.ndarray, action: np.ndarray
    ) -> dict[str, float]:
        pass


class PerformanceAccumulatorWrapper(PerformanceAccumulator):
    def __init__(self, *args: PerformanceAccumulator) -> None:
        self._accumulators = args

    def update_cost(
        self, state: NDArray, reference: NDArray, action: NDArray
    ) -> dict[str, float]:
        return_dict = {}
        for accumulator in self._accumulators:
            return_dict.update(accumulator.update_cost(state, reference, action))

        return return_dict

    @property
    def cost(self):
        return_dict = {}
        for accumulator in self._accumulators:
            return_dict.update(accumulator.cost)

        return return_dict

    @cost.setter
    def cost(self, val):
        for accumulator in self._accumulators:
            accumulator.cost = val


class DeePCCostAccumulator(PerformanceAccumulator):
    def __init__(self, deepc_cost: DeePCCost):
        self._deepc_cost = deepc_cost
        self._cost = 0

    def update_cost(self, state, reference, action):
        delta_state = state - reference
        state_cost = delta_state.T @ self._deepc_cost.Q @ delta_state
        input_cost = action.T @ self._deepc_cost.R @ action

        self._cost += 1e-6 * (state_cost + input_cost)
        return {"cost": self._cost}

    @property
    def cost(self):
        return {"cost": self._cost}

    @cost.setter
    def cost(self, val):
        self._cost = val


class IntegralSquareError(PerformanceAccumulator):
    def __init__(self, mask=None):
        """Computes Integral square error between y and y_ref"""
        self._cost = 0
        self._mask = mask if mask is not None else 1.0

    def update_cost(self, state, reference, action):
        state_cost = np.linalg.norm(self._mask * (state - reference), 2)

        self._cost += 1e-6 * state_cost
        return {"ISE": self._cost}

    @property
    def cost(self):
        return {"ISE": self._cost}

    @cost.setter
    def cost(self, val):
        self._cost = val


class IntegralAbsoluteError:
    def __init__(self, mask=None):
        """Computes Integral absolute error between y and y_ref"""
        self._cost = 0
        self._mask = mask if mask is not None else 1.0

    def update_cost(self, state, reference, action):
        # print(state)
        # print(reference)
        state_cost = np.linalg.norm(self._mask * (state - reference), 1)
        # print(state_cost)

        self._cost += 1e-6 * state_cost
        return {"IAE": self._cost}

    @property
    def cost(self):
        return {"IAE": self._cost}

    @cost.setter
    def cost(self, val):
        self._cost = val


class InputPerformanceAccumulator:
    """Computes some performance metrics for given inputs"""

    def __init__(self, env):
        self.actuator_cost = np.zeros(3)

        main_engine_limits = env.cfg.main_engine_thrust_limits
        side_engine_limits = env.cfg.side_engine_thrust_limits
        nozzle_angle_limits = env.cfg.nozzle_angle_limits

        self._actuation_min = np.array(
            [main_engine_limits[0], side_engine_limits[0], nozzle_angle_limits[0]]
        )
        self._actuation_max = np.array(
            [main_engine_limits[1], side_engine_limits[1], nozzle_angle_limits[1]]
        )

    def update_actuation_cost(self, action, sampling_time=1 / 60):
        action = action.clip(self._actuation_min, self._actuation_max)
        self.actuator_cost += sampling_time * np.abs(action)


class RecursiveAverager:
    """Computes a recursive average without needing to keep track of past data."""

    def __init__(self):
        self._n = 0
        self._avg = 0

    def update(self, x):
        """Update recursive average with a new value"""
        self._n += 1
        self._avg = ((self._n - 1) * self._avg + x) / self._n
        return self._avg

    def __repr__(self):
        return f"avg: {self._avg}"


def show_video(index: int = None, folder: str = "mt"):
    """play back the video"""
    mp4list = glob.glob(f"video/{folder}/*.mp4")
    mp4list.sort()  # index in name should correspond do position in list
    if len(mp4list) > 0:
        vid_index = index if index is not None else 0
        mp4 = mp4list[vid_index]
        video = io.open(mp4, "r+b").read()
        encoded = base64.b64encode(video)
        ipythondisplay.display(
            HTML(
                data="""<video alt="test" autoplay
                loop controls style="height: 400px;">
                <source src="data:video/mp4;base64,{0}" type="video/mp4" />
              </video>""".format(
                    encoded.decode("ascii")
                )
            )
        )
    else:
        print("Could not find video")


def get_simulator_environment(
    initial_condition: np.ndarray = np.array([0.2, 0.9, 0.1]),
    video_folder: str = "video",
    video_title: str = "-1_default_title",
    max_iter=1000,
    **kwargs,
) -> gym.Env:
    """make environment and wrap video so that we can replay them later"""
    initial_condition = np.array(initial_condition)
    if "random_initial_position" in kwargs.keys():
        args = {**kwargs}
    elif initial_condition.size == 3:
        args = {"initial_position": initial_condition, **kwargs}
    elif initial_condition.size == 6:
        args = {"initial_state": initial_condition, **kwargs}
    else:
        raise ValueError(
            "initial condition should be either 3 or 6 dimensional."
            f" Got {initial_condition}"
        )

    env = gym.make(
        "coco_rocket_lander/RocketLander-v0",
        render_mode="rgb_array",
        args=args,
        max_episode_steps=max_iter,
    )
    env = gym.wrappers.RecordVideo(
        env,
        f"video/{video_folder}",
        episode_trigger=lambda x: True,
        name_prefix=video_title,
    )

    # env = TimeDownSampler(env, 2)
    return env


def get_reacher_simulator(
    num_steps=100,
    video_folder: str = os.path.join("video", "reacher"),
    video_title: str = "experiment",
):
    from gymnasium.envs.registration import register

    register(
        id="Reacher-v4-custom",
        entry_point="gymnasium.envs.mujoco.reacher_v4:ReacherEnv",
        max_episode_steps=num_steps,
        reward_threshold=-3.75,
    )
    env = gym.make(
        "Reacher-v4-custom",
        render_mode="rgb_array",
    )
    env = gym.wrappers.RecordVideo(
        env,
        video_folder=video_folder,
        episode_trigger=lambda x: True,
        name_prefix=video_title,
    )
    return env


class TimeDownSampler(gym.Wrapper):
    """Gym Environment Wrapper which downsamples the environment by a given factor

    I.e. if the underlying environment is running at 60 Hz and we downsample it by a factor
    of 3, then the resulting environment is running at 20 Hz.
    """

    def __init__(
        self,
        env: gym.Env,
        downsample_factor: int = 1,
    ):
        super().__init__(env)
        self._downsample_factor = downsample_factor
        self._last_action = None

    def step(self, action):
        """Zero order hold the last input for the predetermined amount of samples"""

        if self._last_action is None:
            obs, reward, terminated, truncated, info = self.env.step(action)
            self._last_action = action
            return obs, reward, terminated, truncated, info

        for _ in range(self._downsample_factor - 1):
            obs, reward, terminated, truncated, info = self.env.step(self._last_action)
            if terminated or truncated:
                return obs, reward, terminated, truncated, info

        obs, reward, terminated, truncated, info = self.env.step(action)
        self._last_action = action
        return obs, reward, terminated, truncated, info


def run_simulator(
    env: gym.Env,
    controller: BaseController,
    setpoint_scheduler: SetPointScheduler = None,
    seed=0,
    deepc_cost_accumulator: Optional[PerformanceAccumulator] = None,
):
    """Runs the simulation for a given env and controller"""
    controller_performance = InputPerformanceAccumulator(env)
    controller_state_trajectory = []
    controller_input_trajectory = []

    can_get_planned_trajectories = hasattr(
        controller, "get_planned_state_trajectory"
    ) and hasattr(controller, "get_planned_input_trajectory")

    controller_state_predictions = [] if can_get_planned_trajectories else None
    controller_input_predictions = [] if can_get_planned_trajectories else None

    simulation_iteration = 0
    obs, info = env.reset(seed=seed)  # specify a random seed for consistency

    controller_execution_times = []

    # simulate
    while True:

        if setpoint_scheduler is not None:
            controller_reference = setpoint_scheduler(obs, env=env)
        else:
            controller_reference = env.get_landing_position()

        # get action
        time_start = perf_counter()
        action = controller.compute_action(obs, controller_reference)
        time_stop = perf_counter()
        controller_execution_times.append(time_stop - time_start)

        if simulation_iteration % 10 == 0 and can_get_planned_trajectories:
            controller_state_predictions.append(
                controller.get_planned_state_trajectory()
            )
            controller_input_predictions.append(
                controller.get_planned_input_trajectory()
            )

        controller_state_trajectory.append(obs[:6])
        controller_input_trajectory.append(action)

        controller_performance.update_actuation_cost(action)
        simulation_iteration += 1

        if deepc_cost_accumulator is not None:
            deepc_cost_accumulator.update_cost(obs[:6], controller_reference, action)

        if action is None:
            print("none action:((")
            break

        next_obs, _, done, truncated, info = env.step(action)

        # check if simulation ended
        if done or truncated:
            print(f"Simulation is done {done} and truncated {truncated}")
            print(f"info:\n{info}")
            if (
                not setpoint_scheduler.is_successful(next_obs)
                and deepc_cost_accumulator is not None
            ):
                deepc_cost_accumulator.cost = np.inf
            break

        # update observation
        obs = next_obs

    env.close()  # video saved at this step

    controller_execution_times = 1000 * np.array(controller_execution_times)
    print(
        f"Average controller execution time: {np.mean(controller_execution_times)}ms, std: {np.std(controller_execution_times)}ms"
    )

    try:
        print(controller._base_controller._solve_time_avg)
    except:
        pass

    controller_state_predictions = np.array(controller_state_predictions)
    controller_input_predictions = np.array(controller_input_predictions)
    controller_state_trajectory = np.array(controller_state_trajectory)
    controller_input_trajectory = np.array(controller_input_trajectory)

    print(f"Total actuation signal needed: {controller_performance.actuator_cost}")

    return (
        controller_state_trajectory,
        controller_input_trajectory,
        controller_state_predictions,
        controller_input_predictions,
        deepc_cost_accumulator.cost if deepc_cost_accumulator is not None else np.nan,
    )


def run_reacher_simulator(
    env: gym.Env,
    controller: BaseController,
    setpoint_scheduler: SetPointScheduler = None,
    seed=0,
    deepc_cost_accumulator: Optional[PerformanceAccumulator] = None,
):
    """Runs the simulation for a given env and controller"""
    controller_state_trajectory = []
    controller_input_trajectory = []

    can_get_planned_trajectories = hasattr(
        controller, "get_planned_state_trajectory"
    ) and hasattr(controller, "get_planned_input_trajectory")

    controller_state_predictions = [] if can_get_planned_trajectories else None
    controller_input_predictions = [] if can_get_planned_trajectories else None

    simulation_iteration = 0
    obs, info = env.reset(seed=seed)  # specify a random seed for consistency

    controller_execution_times = []

    # simulate
    while True:

        if setpoint_scheduler is not None:
            controller_reference = setpoint_scheduler(obs, env=env)
        else:
            assert False

        # get action
        time_start = perf_counter()
        obs_transformed = obs.copy()
        obs_transformed[4:6] += obs[8:10]
        obs_transformed = obs_transformed[[0, 1, 2, 3, 4, 5, 6, 7]]
        action = controller.compute_action(obs_transformed, controller_reference)
        time_stop = perf_counter()
        controller_execution_times.append(time_stop - time_start)

        if simulation_iteration % 5 == 0 and can_get_planned_trajectories:
            controller_state_predictions.append(
                controller.get_planned_state_trajectory()
            )
            controller_input_predictions.append(
                controller.get_planned_input_trajectory()
            )

        controller_state_trajectory.append(obs)
        controller_input_trajectory.append(action)

        simulation_iteration += 1

        if deepc_cost_accumulator is not None:
            deepc_cost_accumulator.update_cost(
                obs_transformed, controller_reference, action
            )

        if action is None:
            print("none action:((")
            break

        next_obs, _, done, truncated, info = env.step(action)

        # check if simulation ended
        if done or truncated:
            print(f"Simulation is done {done} and truncated {truncated}")
            print(f"info:\n{info}")
            if (
                not setpoint_scheduler.is_successful(next_obs)
                and deepc_cost_accumulator is not None
            ):
                deepc_cost_accumulator.cost = np.inf
            break

        # update observation
        obs = next_obs

    env.close()  # video saved at this step

    controller_execution_times = 1000 * np.array(controller_execution_times)
    print(
        f"Average controller execution time: {np.mean(controller_execution_times)}ms, std: {np.std(controller_execution_times)}ms"
    )

    controller_state_predictions = np.array(controller_state_predictions)
    controller_input_predictions = np.array(controller_input_predictions)
    controller_state_trajectory = np.array(controller_state_trajectory)
    controller_input_trajectory = np.array(controller_input_trajectory)

    return (
        controller_state_trajectory,
        controller_input_trajectory,
        controller_state_predictions,
        controller_input_predictions,
        deepc_cost_accumulator.cost if deepc_cost_accumulator is not None else np.nan,
        controller._solve_time_avg,
    )


def plot_trajectory(
    x_traj, x_traj_predictions, plot_heading=True, shift_preds=False, filename=None
):
    fig = plt.figure(dpi=200)
    plt.title("DeePC prediction vs actual trajectory")
    plt.plot(x_traj[:, 0], x_traj[:, 1], label="actual rocket trajectory", c="blue")

    # Plot DeePC predictions
    for i, traj in enumerate(x_traj_predictions):
        if shift_preds:
            x_init = x_traj[10 * i, :2]
            delta_x_init = x_init - traj[0, :2]
            traj[:, :2] += delta_x_init
        try:
            if i == 0:
                plt.plot(traj[:, 0], traj[:, 1], "--", label="predicted trajectories")
            else:
                plt.plot(traj[:, 0], traj[:, 1], "--")

            if plot_heading:
                for step in range(traj.shape[0]):
                    plt.plot(
                        [traj[step, 0], traj[step, 0] + np.sin(-traj[step, 4])],
                        [traj[step, 1], traj[step, 1] + np.cos(-traj[step, 4])],
                        c="black",
                        linewidth=0.1,
                    )
        except Exception as e:
            print(e)

    # Plot rocket heading. Always plot first heading.
    for step in range(x_traj.shape[0]):
        plt.plot(
            [x_traj[step, 0], x_traj[step, 0] + np.sin(-x_traj[step, 4])],
            [x_traj[step, 1], x_traj[step, 1] + np.cos(-x_traj[step, 4])],
            c="red",
            linewidth=0.1,
        )
        if not plot_heading:
            break

    plt.xlim(0, 30)
    plt.ylim(5, 28)
    plt.legend()
    if filename is not None:
        plt.savefig(filename)

    plt.show()


def load_data_from_folder(
    folder: str, dataset_name: Optional[str] = None
) -> TrajectoryDataSet:
    """Load all the data from a specific directory and return the state and input data
    as arrays in a list. I.e. if you index the state list and input list with the same idx,
    you will get the corresponding trajectories.
    """
    files = [
        file
        for file in os.listdir(folder)
        if os.path.isfile(os.path.join(folder, file))
    ]
    state_files = [file for file in files if "state" in file]
    input_files = [file for file in files if "input" in file]
    assert len(state_files) + len(input_files) == len(files)
    state_files = sorted(state_files)
    input_files = sorted(input_files)
    trajectories = []
    for state_file, input_file in zip(state_files, input_files):
        state_data = np.atleast_2d(
            np.loadtxt(os.path.join(folder, state_file), delimiter=",")
        )
        input_data = np.atleast_2d(
            np.loadtxt(os.path.join(folder, input_file), delimiter=",")
        )
        trajectories.append(TrajectoryData(state_data, input_data))

    return TrajectoryDataSet(
        trajectories, dataset_name if dataset_name is not None else folder
    )


def compute_dataset_angles(H: NDArray) -> NDArray:
    """Computes distrubtion of angles between datapoints in H

    Uses $a\cdot b = \|a\|\|b\|\cos\alpha$.
    """

    norm_data = H / np.linalg.norm(H, axis=0, keepdims=True)
    angles = np.arccos(np.dot(norm_data.T, norm_data))
    # Get upper triangular indices cuz symmetric. (also ignore diag since angle is always 90°)
    i_upper = np.triu_indices_from(angles, k=1)
    return angles[i_upper], angles
