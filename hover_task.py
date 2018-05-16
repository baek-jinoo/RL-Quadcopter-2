import numpy as np
from physics_sim import PhysicsSim
from collections import deque

class HoverTask():
    """Task (environment) that defines the goal and provides feedback to the agent."""
    def __init__(self, init_pose=None, init_velocities=None, 
        init_angle_velocities=None, runtime=5., target_state=None):
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
        self.sim = PhysicsSim(init_pose, init_velocities, init_angle_velocities, runtime)
        self.action_repeat = 3

        self.single_state_size = 16
        self.state_size = self.action_repeat * self.single_state_size
        self.action_low = 0
        self.action_high = 900
        self.action_size = 4
        self.actions = [0, 0, 0, 0]

        # Goal
        self.target_state = target_state if target_state is not None else np.array([0., 0., 10.])

    def get_current_state(self):
        return np.array(list(self.sim.pose) + list(self.sim.v) + list(self.sim.angular_v) + list(self.actions))

    def get_reward(self):
        """Uses current pose of sim to return reward."""
        current_state_for_reward = np.array(list(self.sim.pose)[:3])
        reward = 0.1-.03*(abs(current_state_for_reward - self.target_state)).sum()
        return reward

    def step(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""
        self.actions = rotor_speeds
        reward = 0
        state_all = []
        for _ in range(self.action_repeat):
            done = self.sim.next_timestep(rotor_speeds) # update the sim pose and velocities
            reward += self.get_reward()
            state_all.append(self.get_current_state())
        next_state = np.concatenate(state_all)
        return next_state, reward, done

    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()
        self.actions = [0, 0, 0, 0]
        state = np.concatenate([self.get_current_state()] * self.action_repeat)
        return state

