import sys
import os
import numpy as np
import csv
import matplotlib.pyplot as plt
import keras
from agents.helper import mv_file_to_dir_with_date

class Runner():
    def __init__(self,
                 task,
                 agent):
        self.task = task
        self.agent = agent
        self.labels = ['time', 'x', 'y', 'z', 'phi', 'theta', 'psi', 'x_velocity',
                       'y_velocity', 'z_velocity', 'phi_velocity', 'theta_velocity',
                       'psi_velocity', 'rotor_speed1', 'rotor_speed2', 'rotor_speed3', 'rotor_speed4', 'reward', 'episode']
        self.labels_per_episode = ['episode', 'mean_reward']

    def run(self,
            runtime=100,
            display_graph=True,
            display_freq=5,
            should_write_results_to_file=False,
            experiences_to_mimic=None,
            results_file_output='data',
            episodic_results_file_output='episodic_data',
            outputs_folder='data_outputs',
            num_episode=10,
            weights_directory='weights_backup',
            load_weights_from_file=False,
            save_weights=False):

        self._setup_figures_for_dynamic_plots()

        results = {x : [] for x in self.labels}
        episode_results = {x: [] for x in self.labels_per_episode}

        max_time_steps = int(runtime)

        if experiences_to_mimic is not None and hasattr(self.agent, "mimic"):
            self.agent.mimic(experiences_to_mimic)

        done = False

        if load_weights_from_file:
            self.agent.load_weights(location=weights_directory)

        self._mv_file_to_dir_with_date(results_file_output, outputs_folder)
        self._mv_file_to_dir_with_date(episodic_results_file_output, outputs_folder)

        with open(results_file_output, 'w') as csvfile, open(episodic_results_file_output, 'w') as episodic_csvfile:
            writer = csv.writer(csvfile)
            episode_writer = csv.writer(episodic_csvfile)
            writer.writerow(self.labels)
            episode_writer.writerow(self.labels_per_episode)
            for i_episode in range(1, num_episode + 1):
                state = self.agent.reset_episode()
                self.task.reset()
                episode_rewards = []
                results_per_episode = {x : [] for x in self.labels}
                for i, t in enumerate(range(max_time_steps)):
                    rotor_speeds = self.agent.act(state)
                    next_state, reward, done = self.task.step(rotor_speeds)
                    self.agent.step(rotor_speeds, reward, next_state, done)

                    step_results = [self.task.sim.time] + list(self.task.sim.pose) + list(self.task.sim.v) + list(self.task.sim.angular_v) + list(rotor_speeds)
                    step_results.append(reward)
                    step_results.append(i_episode)
                    for ii in range(len(self.labels)):
                        results[self.labels[ii]].append(step_results[ii])
                        results_per_episode[self.labels[ii]].append(step_results[ii])
                    self._write(step_results, writer, should_write_results_to_file)

                    episode_rewards.append(reward)

                    state = next_state

                    if done or i == max_time_steps-1:
                        episode_step_result = [i_episode, np.mean(episode_rewards)]
                        for ii in range(len(self.labels_per_episode)):
                            episode_results[self.labels_per_episode[ii]].append(episode_step_result[ii])
                        self._write(episode_step_result, episode_writer, should_write_results_to_file)
                        if display_graph:
                            self._plt_dynamic_reward(results)
                            self._plt_dynamic_reward_means(episode_results)
                            self._plt_dynamic_x_y_z(results_per_episode)
                            self._plt_dynamic_rotors(results_per_episode)
                        break
                    else:
                        if t % display_freq == 0 and display_graph:
                            self._plt_dynamic_reward(results)
        self._mv_file_to_dir_with_date(results_file_output, outputs_folder)
        self._mv_file_to_dir_with_date(episodic_results_file_output, outputs_folder)

        if save_weights:
            self.agent.save_weights(location=weights_directory)

    def _mv_file_to_dir_with_date(self, filename, outputs_folder):
        cwd = os.getcwd()
        origin_file_path = os.path.join(cwd, filename)

        destination_path = os.path.join(cwd, outputs_folder)

        mv_file_to_dir_with_date(origin_file_path, destination_path)

    #def _mv_to_file_with_date(self, filename, outputs_folder):
    #    cwd = os.getcwd()
    #    origin_file_path = os.path.join(cwd, filename)

    #    if not os.path.isfile(origin_file_path):
    #        return

    #    destination_directory = os.path.join(cwd, outputs_folder)

    #    if not os.path.exists(destination_directory):
    #        os.makedirs(destination_directory)

    #    timestr = time.strftime("%Y%m%d-%H%M%S")
    #    destination_filename = timestr + '-' + filename + '.csv'
    #    destination_file_path = os.path.join(destination_directory, destination_filename)

    #    shutil.move(origin_file_path,
    #                destination_file_path)

    def _setup_figures_for_dynamic_plots(self):
        fig1, (ax11, ax12, ax_x, ax_rotors) = plt.subplots(4, 1)

        ax11.set_title("Rewards")
        ax12.set_title("Average rewards")
        ax_x.set_title("x, y, z")
        ax_rotors.set_title("rotor speeds")

        fig1.tight_layout(pad=4.0, w_pad=1.0, h_pad=0.1)
        fig1.set_size_inches(4, 6)
        fig1.show()

        self.fig1 = fig1
        self.ax_rotors = ax_rotors
        self.ax_x = ax_x
        self.ax11 = ax11
        self.ax12 = ax12

    def _plt_dynamic_reward(self, results):
        self.ax11.plot(results['reward'])
        self.fig1.canvas.draw()

    def _plt_dynamic_reward_means(self, episode_results):
        self.ax12.plot(episode_results['mean_reward'])
        self.fig1.canvas.draw()

    def _plt_dynamic_x_y_z(self, results_per_episode):
        self.ax_x.clear()
        self.ax_x.plot(results_per_episode['time'], results_per_episode['x'], label='x', color='green')
        self.ax_x.plot(results_per_episode['time'], results_per_episode['y'], label='y', color='red')
        self.ax_x.plot(results_per_episode['time'], results_per_episode['z'], label='z', color='blue')
        self.fig1.canvas.draw()

    def _plt_dynamic_rotors(self, results_per_episode):
        self.ax_rotors.clear()
        self.ax_rotors.plot(results_per_episode['time'], results_per_episode['rotor_speed1'], label='1', color='green')
        self.ax_rotors.plot(results_per_episode['time'], results_per_episode['rotor_speed2'], label='2', color='red')
        self.ax_rotors.plot(results_per_episode['time'], results_per_episode['rotor_speed3'], label='3', color='blue')
        self.ax_rotors.plot(results_per_episode['time'], results_per_episode['rotor_speed4'], label='4', color='magenta')
        self.fig1.canvas.draw()

    def _write(self, step_results, writer, should_write_results_to_file=False):
        if should_write_results_to_file:
            writer.writerow(step_results)

