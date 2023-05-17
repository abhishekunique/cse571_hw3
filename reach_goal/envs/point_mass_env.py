import math
import gym
import numpy as np
import pybullet as p

from reach_goal.resources.plane import Plane
from reach_goal.resources.point_mass import PointMass
from reach_goal.resources.goal import Goal

class PointMassEnv(gym.Env):
  metadata = {'render.modes': ['human']}
  """Trains an agent to go fast."""
  def __init__(self, render):
    self.action_space = gym.spaces.box.Box(
        low=np.array([-1.0, -1.0], dtype=np.float32),
        high=np.array([1.0, 1.0], dtype=np.float32))
    self.observation_space = gym.spaces.box.Box(
        low=np.array([-10, -10, -10, -10], dtype=np.float32),
        high=np.array([10, 10, 10, 10], dtype=np.float32))
    # THe self.observation - [point_mass.x, point_mass.y, goal.x, goal.y]
    self.np_random, _ = gym.utils.seeding.np_random()

    if render:
      self.client = p.connect(p.GUI)
    else:
      self.client = p.connect(p.DIRECT)
    # Reduce length of episodes for RL algorithms
    self.dt = 1/30
    p.setTimeStep(self.dt, self.client)

    self.point_mass = None
    self.goal = None
    self.done = False
    self.prev_dist_to_goal = None
    self.rendered_img = None
    self.render_rot_matrix = None
    self.reset()

  def reset(self):
    p.resetSimulation(self.client)
    # p.setGravity(0, 0, -10)
    # Reload the plane and car
    Plane(self.client)
    self.point_mass = PointMass(self.dt, self.client)

    # Set the goal to a random target
    x = (self.np_random.uniform(5, 9) if self.np_random.randint(2) else
            self.np_random.uniform(-5, -9))
    y = (self.np_random.uniform(5, 9) if self.np_random.randint(2) else
            self.np_random.uniform(-5, -9))
    self.goal = (9, 9)
    self.done = False

    # Visual element of the goal
    Goal(self.client, self.goal)

    # Get observation to return
    point_mass_ob = self.point_mass.get_observation()

    self.prev_dist_to_goal = math.sqrt(((point_mass_ob[0] - self.goal[0]) ** 2 +
                                        (point_mass_ob[1] - self.goal[1]) ** 2))

    # Check the return value
    return np.array(point_mass_ob + self.goal, dtype=np.float32)

  def seed(self, seed=None):
      self.np_random, seed = gym.utils.seeding.np_random(seed)
      return [seed]

  def step(self, action):
    # Feed action to the car and get observation of car's state
    # TODO: Apply action cutoff
    self.point_mass.apply_action(action)
    p.stepSimulation()
    point_mass_ob = self.point_mass.get_observation()

    # Compute reward as L2 change in distance to goal
    dist_to_goal = math.sqrt(((point_mass_ob[0] - self.goal[0]) ** 2 +
                              (point_mass_ob[1] - self.goal[1]) ** 2))
    # The max dist is (20**2+20**2), therefore normalizing
    reward = -dist_to_goal/800

    # Done by running off boundaries
    if (point_mass_ob[0] >= 15 or point_mass_ob[0] <= -15 or
            point_mass_ob[1] >= 15 or point_mass_ob[1] <= -15):
        self.done = True
    
    # Done by reaching goal
    elif dist_to_goal < 1:
        self.done = True
        reward = 5

    ob = np.array(point_mass_ob + self.goal, dtype=np.float32)
    return ob, reward, self.done, dict()

  # @property
  # def observation_size(self):
  #   return 2

  # @property
  # def action_size(self):
  #   return 2