"""Continuous control variant of highway-v0."""
import numpy as np
from gym import spaces
from highway_env import utils
from highway_env.envs.highway_env import AbstractEnv
from highway_env.envs.highway_env import HighwayEnv
from highway_env.vehicle.control import MDPVehicle
from highway_env.vehicle.dynamics import Vehicle
from ray.rllib.utils.annotations import override


class HighwayContinuousEnv(HighwayEnv):
    """Subclass of HighwayEnv that takes vectors in [-1, 1]^2 as actions."""

    STEERING_RANGE = np.pi / 4
    ACCELERATION_RANGE = 5.0

    @override(AbstractEnv)
    def __init__(self):
        super().__init__()
        self._old_lane = None

    @override(AbstractEnv)
    def define_spaces(self):
        super().define_spaces()
        self.action_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)

    @override(HighwayEnv)
    def reset(self):
        obs = super().reset()
        self._old_lane = self._curr_lane()
        return obs

    @override(HighwayEnv)
    def _create_vehicles(self):
        self.vehicle = Vehicle.create_random(self.road, 0)
        self.road.vehicles.append(self.vehicle)

        vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])
        for _ in range(self.config["vehicles_count"]):
            self.road.vehicles.append(vehicles_type.create_random(self.road))

    @override(HighwayEnv)
    def _reward(self, _):
        idx = self._curr_lane()
        action_reward = self.LANE_CHANGE_REWARD if idx != self._old_lane else 0
        self._old_lane = idx
        neighbours = self.road.network.all_side_lanes(self.vehicle.lane_index)
        state_reward = (
            +self.config["collision_reward"] * self.vehicle.crashed
            + self.RIGHT_LANE_REWARD
            * self.vehicle.lane_index[2]
            / (len(neighbours) - 1)
            + self.HIGH_VELOCITY_REWARD
            * MDPVehicle.speed_to_index(self.vehicle.velocity)
            / (MDPVehicle.SPEED_COUNT - 1)
        )
        return utils.remap(
            action_reward + state_reward,
            [
                self.config["collision_reward"],
                self.HIGH_VELOCITY_REWARD + self.RIGHT_LANE_REWARD,
            ],
            [0, 1],
        )

    @override(HighwayEnv)
    def _is_terminal(self):
        vehicle = self.vehicle
        return super()._is_terminal() or not vehicle.lane.on_lane(vehicle.position)

    @override(AbstractEnv)
    def _simulate(self, action=None):
        for _ in range(
            int(self.SIMULATION_FREQUENCY // self.config["policy_frequency"])
        ):
            if (
                action is not None
                and self.time
                % int(self.SIMULATION_FREQUENCY // self.config["policy_frequency"])
                == 0
            ):
                # Forward action to the vehicle
                self.vehicle.act(
                    {
                        "acceleration": action[0] * self.ACCELERATION_RANGE,
                        "steering": action[1] * self.STEERING_RANGE,
                    }
                )

            self.road.act()
            self.road.step(1 / self.SIMULATION_FREQUENCY)
            self.time += 1

            # Automatically render intermediate simulation steps if a viewer has been
            # launched. Ignored if the rendering is done offscreen
            self._automatic_rendering()

            # Stop at terminal states
            if self.done or self._is_terminal():
                break
        self.enable_auto_render = False

    def _curr_lane(self):
        _, _, idx = self.vehicle.lane_index
        return idx
