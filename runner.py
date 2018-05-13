import sys
import numpy as np
import csv

import matplotlib.pyplot as plt
plt.rcParams['figure.dpi'] = 150


def setup_figures_for_dynamic_plots():
    fig1, (ax11, ax12, ax_x, ax_rotors) = plt.subplots(4, 1)

    ax11.set_title("Rewards")
    ax12.set_title("Average rewards")
    ax_x.set_title("x, y, z")
    ax_rotors.set_title("rotor speeds")

    fig1.tight_layout(pad=4.0, w_pad=1.0, h_pad=0.1)
    fig1.set_size_inches(4, 8)
    fig1.show()

def plt_dynamic_reward(rewards):
    ax11.plot(rewards)
    fig1.canvas.draw()

def plt_dynamic_reward_means(reward_means):
    ax12.plot(reward_means)
    fig1.canvas.draw()

def plt_dynamic_x_y_z(results_per_episode):
    ax_x.clear()
    ax_x.plot(results_per_episode['time'], results_per_episode['x'], label='x', color='green')
    ax_x.plot(results_per_episode['time'], results_per_episode['y'], label='y', color='red')
    ax_x.plot(results_per_episode['time'], results_per_episode['z'], label='z', color='blue')
    fig1.canvas.draw()

def plt_dynamic_rotors(results_per_episode):
    ax_rotors.clear()
    ax_rotors.plot(results_per_episode['time'], results_per_episode['rotor_speed1'], label='1', color='green')
    ax_rotors.plot(results_per_episode['time'], results_per_episode['rotor_speed2'], label='2', color='red')
    ax_rotors.plot(results_per_episode['time'], results_per_episode['rotor_speed3'], label='3', color='blue')
    ax_rotors.plot(results_per_episode['time'], results_per_episode['rotor_speed4'], label='4', color='magenta')
    fig1.canvas.draw()


# Modify the values below to give the quadcopter a different starting position.
#runtime = 100.                                     # time limit of the episode
#init_pose = np.array([0., 0., 0., 0., 0., 0.])  # initial pose
#init_velocities = np.array([0., 0., 0.])         # initial velocities
#init_angle_velocities = np.array([0., 0., 0.])   # initial angle velocities
#file_output = 'data.txt'                         # file name for saved results
#    
#num_episode = 800
#
## general configuration
#display_graph = True
#display_freq = 5
#
#task = HoverTask(init_pose, init_velocities, init_angle_velocities, runtime)
#agent = DDPG(task)

def write(labels, step_results, results):
    for ii in range(len(labels)):
        results[labels[ii]].append(step_results[ii])
    if should_write:
        writer.writerow(step_results)

def runner(task,
           agent,
           runtime=100,
           display_graph=True,
           display_freq=5,
           should_write=False):
    done = False

    labels = ['time', 'x', 'y', 'z', 'phi', 'theta', 'psi', 'x_velocity',
              'y_velocity', 'z_velocity', 'phi_velocity', 'theta_velocity',
              'psi_velocity', 'rotor_speed1', 'rotor_speed2', 'rotor_speed3', 'rotor_speed4', 'reward', 'episode']
    results = {x : [] for x in labels}
    labels_per_episode = ['episode', 'mean_reward']
    results_per_episode = {x: [] for x in labels_per_episode}
    rewards = []
    reward_means = []

    max_time_steps = int(runtime)

    # experiences_to_mimic = create_mimic_experiences()
    agent.mimic(experiences_to_mimic)

    with open(file_output, 'w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(labels)
        for i_episode in range(1, num_episode + 1):
            state = agent.reset_episode()
            task.reset()
            episode_rewards = []
            results_per_episode = {x : [] for x in labels}
            for i, t in enumerate(range(max_time_steps)):
                rotor_speeds = agent.act(state)
                next_state, reward, done = task.step(rotor_speeds)
                agent.step(rotor_speeds, reward, next_state, done)

                step_results = [task.sim.time] + list(task.sim.pose) + list(task.sim.v) + list(task.sim.angular_v) + list(rotor_speeds)
                step_results.append(reward)
                step_results.append(i_episode)
                for ii in range(len(labels)):
                    results[labels[ii]].append(step_results[ii])
                if should_write:
                    writer.writerow(step_results)

                rewards.append(reward)
                episode_rewards.append(reward)

                state = next_state

                if done or i == max_time_steps-1:
                    if display_graph:
                        plt_dynamic_reward(rewards)
                        reward_means.append(np.mean(episode_rewards))
                        plt_dynamic_x_y_z(results_per_episode)
                        plt_dynamic_reward_means(reward_means)
                        plt_dynamic_rotors(results_per_episode)
                    break
                else:
                    if t % display_freq == 0:
                        plt_dynamic_reward(rewards)
