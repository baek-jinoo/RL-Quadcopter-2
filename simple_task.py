import gym
import numpy as np
from collections import deque

class SimpleTask():
    """Task (environment) that defines the goal and provides feedback to the agent."""
    def __init__(self):
        """Initialize a Task object.
        Params
        ======
            init_pose: initial position of the quadcopter in (x,y,z) dimensions and the Euler angles
            init_velocities: initial velocity of the quadcopter in (x,y,z) dimensions
            init_angle_velocities: initial radians/second for each of the three Euler angles
            runtime: time limit for each episode
            target_state: target/goal (x,y,z, 'phi', 'theta', 'psi', 'x_velocity',
            'y_velocity', 'z_velocity', 'phi_velocity', 'theta_velocity',
            'psi_velocity') state for the agent
        """
        # Simulation
        self.env = gym.make('MountainCarContinuous-v0')

        self.action_repeat = 3

        self.single_state_size = 2
        self.state_size = self.action_repeat * self.single_state_size
        self.action_low = -1.
        self.action_high = 1.
        self.action_size = 1
        self.actions = [0.]

    def step(self, action):
        """Uses action to obtain next state, reward, done."""
        self.actions = action
        state_all = []
        cum_reward = 0
        done = False
        for i in range(self.action_repeat):
            state, reward, done, _ = self.env.step(action)
            cum_reward += reward
            state_all.append(state)
            if done:
                remaining = self.action_repeat - (i + 1)
                for ii in range(remaining):
                    cum_reward += reward
                    state_all.append(state)
                break
        next_state = np.concatenate(state_all)
        return next_state, cum_reward, done

    def reset(self):
        """Reset the sim to start a new episode."""
        self.actions = [0.]
        return np.concatenate([self.env.reset()] * self.action_repeat)

