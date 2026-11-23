"""
CartPole obstacle-avoiding orbit app (Webots R2025a + deepbots, Robot-Supervisor scheme).

The CartPole robot drives in counter-clockwise circles around a target object
(the yellow cylinder, DEF TARGET) at a desired radius, while five front distance
sensors let it steer around obstacles that sit on its path.

Structure (same pattern as the deepbots CartPole tutorial):
    CartpoleNavEnv  -> deepbots RobotSupervisorEnv (gym-style env: observations,
                       reward, done, apply_action)
    OrbitAvoidAgent -> a simple rule-based policy with the same .work(obs) interface
                       as the tutorial's PPOAgent, so it can be swapped for an RL agent.
"""
import math

import numpy as np
from gym.spaces import Box
from deepbots.supervisor.controllers.robot_supervisor_env import RobotSupervisorEnv

from utilities import normalize_to_range, wrap_angle

# ------------------------------ parameters ---------------------------------
ORBIT_RADIUS = 1.0          # desired distance from the target centre [m]
MAX_WHEEL_SPEED = 10.0      # rad/s  (wheel radius 0.04 m -> 0.4 m/s)
SENSOR_NAMES = ["ds_left", "ds_front_left", "ds_front", "ds_front_right", "ds_right"]
LAPS_PER_EPISODE = 3
MAX_STEPS_PER_EPISODE = 6000
COLLISION_THRESHOLD = 0.93  # normalised proximity (1.0 = touching)


class CartpoleNavEnv(RobotSupervisorEnv):
    def __init__(self):
        super().__init__()
        # obs = [radius error, heading error, 5 x proximity, pole angle]
        self.observation_space = Box(low=np.array([-1.0, -1.0] + [0.0] * 5 + [-1.0]),
                                     high=np.array([1.0, 1.0] + [1.0] * 5 + [1.0]),
                                     dtype=np.float64)
        # action = [left wheel speed, right wheel speed] in [-1, 1]
        self.action_space = Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float64)

        self.robot = self.getSelf()
        self.target = self.getFromDef("TARGET")
        # Supervisor-only: obstacle nodes used for logging true clearance (not given to the agent)
        self.obstacles = [self.getFromDef(f"OBSTACLE_{i}") for i in (1, 2, 3)]
        self.min_clearance = float("inf")

        self.sensors = []
        for name in SENSOR_NAMES:
            ds = self.getDevice(name)
            ds.enable(self.timestep)
            self.sensors.append(ds)

        self.pole_sensor = self.getDevice("polePosSensor")
        self.pole_sensor.enable(self.timestep)

        # wheel1/wheel2 are on the left (+y), wheel3/wheel4 on the right (-y)
        self.left_wheels, self.right_wheels = [], []
        for name in ["wheel1", "wheel2", "wheel3", "wheel4"]:
            motor = self.getDevice(name)
            motor.setPosition(float("inf"))
            motor.setVelocity(0.0)
            (self.left_wheels if name in ("wheel1", "wheel2") else self.right_wheels).append(motor)

        self.steps = 0
        self.prev_phi = None
        self.travelled_angle = 0.0
        self.collisions = 0
        self.episode_score = 0.0

    # ----------------------------- helpers ---------------------------------
    def _pose(self):
        x, y, _ = self.robot.getPosition()
        rot = self.robot.getOrientation()          # 3x3 row-major
        yaw = math.atan2(rot[3], rot[0])
        tx, ty, _ = self.target.getPosition()
        return x - tx, y - ty, yaw

    def _proximities(self):
        # DistanceSensor lookup table maps 0 m -> 1000, 0.6 m -> 0
        return [min(max(s.getValue() / 1000.0, 0.0), 1.0) for s in self.sensors]

    # ------------------------- deepbots interface --------------------------
    def get_observations(self):
        dx, dy, yaw = self._pose()
        r = math.hypot(dx, dy)
        phi = math.atan2(dy, dx)

        # Desired heading for a CCW orbit: tangent + correction toward the circle
        desired = phi + math.pi / 2 + math.atan(2.0 * (r - ORBIT_RADIUS))
        heading_error = wrap_angle(desired - yaw)

        return ([normalize_to_range(r - ORBIT_RADIUS, -1.0, 1.0, -1.0, 1.0, clip=True),
                 heading_error / math.pi]
                + self._proximities()
                + [normalize_to_range(self.pole_sensor.getValue(), -0.6, 0.6, -1.0, 1.0, clip=True)])

    def get_default_observation(self):
        return [0.0] * self.observation_space.shape[0]

    def get_reward(self, action=None):
        dx, dy, _ = self._pose()
        phi = math.atan2(dy, dx)
        progress = 0.0 if self.prev_phi is None else wrap_angle(phi - self.prev_phi)
        self.prev_phi = phi
        self.travelled_angle += progress

        reward = 10.0 * progress                                    # move CCW around target
        reward -= 0.05 * abs(math.hypot(dx, dy) - ORBIT_RADIUS)    # stay near the orbit
        if max(self._proximities()) > COLLISION_THRESHOLD:          # hit / brushed obstacle
            reward -= 5.0
            self.collisions += 1
        return reward

    def is_done(self):
        return (self.travelled_angle >= LAPS_PER_EPISODE * 2 * math.pi
                or self.steps >= MAX_STEPS_PER_EPISODE)

    def get_info(self):
        x, y, _ = self.robot.getPosition()
        for obs in self.obstacles:
            ox, oy, _ = obs.getPosition()
            self.min_clearance = min(self.min_clearance, math.hypot(x - ox, y - oy))
        return {"laps": self.travelled_angle / (2 * math.pi), "collisions": self.collisions,
                "closest_obstacle": self.min_clearance}

    def render(self, mode="human"):
        pass

    def apply_action(self, action):
        left, right = np.clip(action, -1.0, 1.0) * MAX_WHEEL_SPEED
        # simulationReset() puts motors back in position mode, so re-enable velocity mode
        for m in self.left_wheels:
            m.setPosition(float("inf"))
            m.setVelocity(float(left))
        for m in self.right_wheels:
            m.setPosition(float("inf"))
            m.setVelocity(float(right))
        self.steps += 1

    def reset(self):
        obs = super().reset()
        self.steps = 0
        self.prev_phi = None
        self.travelled_angle = 0.0
        self.collisions = 0
        self.episode_score = 0.0
        self.min_clearance = float("inf")
        return obs


class OrbitAvoidAgent:
    """Rule-based policy: follow the orbit vector field, blend in obstacle avoidance."""

    def __init__(self, cruise=0.7, k_heading=2.5, k_avoid=4.0):
        self.cruise = cruise
        self.k_heading = k_heading
        self.k_avoid = k_avoid

    def work(self, obs):
        _, heading_err, left, front_left, front, front_right, right, _ = obs

        # 1) Orbit-following: turn toward the desired heading
        turn = self.k_heading * heading_err * math.pi / 2

        # 2) Obstacle avoidance (Braitenberg style). The target is always on the left
        #    in a CCW orbit, so an obstacle straight ahead is passed on the outside (right).
        danger = max(front_left, front, front_right)
        avoid = (front_right + 0.5 * right) - (front_left + 0.5 * left) - 1.2 * front
        turn = (1.0 - danger) * turn + self.k_avoid * avoid

        # 3) Slow down when something is close in front
        speed = self.cruise * (1.0 - 0.7 * min(1.0, front + 0.5 * max(front_left, front_right)))

        left_cmd = float(np.clip(speed - turn * 0.5, -1.0, 1.0))
        right_cmd = float(np.clip(speed + turn * 0.5, -1.0, 1.0))
        return [left_cmd, right_cmd]


if __name__ == "__main__":
    env = CartpoleNavEnv()
    agent = OrbitAvoidAgent()

    episode = 0
    while True:
        observation = env.reset()
        done = False
        while not done:
            action = agent.work(observation)
            observation, reward, done, info = env.step(action)
            env.episode_score += reward
            if env.steps % 250 == 0:
                print(f"step {env.steps:5d} | laps {info['laps']:.2f} | "
                      f"collisions {info['collisions']} | score {env.episode_score:.1f}")
        print(f"Episode #{episode} finished: laps={info['laps']:.2f}, "
              f"collisions={info['collisions']}, score={env.episode_score:.1f}, "
              f"closest obstacle centre={info['closest_obstacle']:.2f} m")
        episode += 1
