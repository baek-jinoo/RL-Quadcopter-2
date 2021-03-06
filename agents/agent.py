from agents.replay_buffer import ReplayBuffer
import numpy as np
from agents.actor import Actor
from agents.critic import Critic
from agents.noise import OUNoise
import h5py
import os
from agents.helper import mv_file_to_dir_with_date

from keras.callbacks import TensorBoard
import keras.callbacks as callbacks
import tensorflow as tf

class DDPG():
    """Reinforcement Learning agent that learns using DDPG."""
    def __init__(self, task, verbose=False):
        self.verbose = verbose

        self.task = task
        self.state_size = task.state_size
        self.action_size = task.action_size
        self.action_low = task.action_low
        self.action_high = task.action_high

        # Actor (Policy) Model
        self.actor_local = Actor(self.state_size, self.action_size, self.action_low, self.action_high)
        self.actor_target = Actor(self.state_size, self.action_size, self.action_low, self.action_high)

        # Critic (Value) Model
        self.critic_local = Critic(self.state_size, self.action_size)
        self.critic_target = Critic(self.state_size, self.action_size)

        #log_path = '/tmp/logs'
        #self.callback = callbacks.TensorBoard(log_dir=log_path, histogram_freq=1,
        #                        write_images=False, write_grads=True, write_graph=False)
        #self.callback.set_model(self.critic_local.model)

        #log_path = '/tmp/logs'
        #self.writer = tf.summary.FileWriter(log_path)

        #self.learn_counter = 0

        # Initialize target model parameters with local model parameters
        self.critic_target.model.set_weights(self.critic_local.model.get_weights())
        self.actor_target.model.set_weights(self.actor_local.model.get_weights())

        # Noise process
        self.exploration_mu = 0.1
        self.exploration_theta = 0.2
        self.exploration_sigma = 0.2
        self.noise = OUNoise(self.action_size, self.exploration_mu, self.exploration_theta, self.exploration_sigma)

        # Replay memory
        self.buffer_size = 100000
        self.batch_size = 512
        self.memory = ReplayBuffer(self.buffer_size, self.batch_size)

        # Algorithm parameters
        self.gamma = 0.99  # discount factor
        self.tau = 0.015  # for soft update of target parameters

    def reset_episode(self):
        self.noise.reset()
        state = self.task.reset()
        self.last_state = state
        #self.learn_counter = 0
        return state

    def mimic(self, experience_to_mimic):
        print("ready to mimic")
        self.memory.memory = experience_to_mimic

    def step(self, action, reward, next_state, done):
         # Save experience / reward
        self.memory.add(self.last_state, action, reward, next_state, done)

        # Learn, if enough samples are available in memory
        if len(self.memory) > self.batch_size:
            experiences = self.memory.sample()
            self.learn(experiences)

        # Roll over last state and action
        self.last_state = next_state

    def act(self, state):
        """Returns actions for given state(s) as per current policy."""
        state = np.reshape(state, [-1, self.state_size])
        action = self.actor_local.model.predict(state)[0]
        return list(action + self.noise.sample())  # add some noise for exploration

    def learn(self, experiences):
        """Update policy and value parameters using given batch of experience tuples."""
        # Convert experience tuples to separate arrays for each element (states, actions, rewards, etc.)
        states = np.vstack([e.state for e in experiences if e is not None])
        actions = np.array([e.action for e in experiences if e is not None]).astype(np.float32).reshape(-1, self.action_size)
        rewards = np.array([e.reward for e in experiences if e is not None]).astype(np.float32).reshape(-1, 1)
        dones = np.array([e.done for e in experiences if e is not None]).astype(np.uint8).reshape(-1, 1)
        next_states = np.vstack([e.next_state for e in experiences if e is not None])

        # Get predicted next-state actions and Q values from target models
        #     Q_targets_next = critic_target(next_state, actor_target(next_state))
        actions_next = self.actor_target.model.predict_on_batch(next_states)
        Q_targets_next = self.critic_target.model.predict_on_batch([next_states, actions_next])

        # Compute Q targets for current states and train critic model (local)
        Q_targets = rewards + self.gamma * Q_targets_next * (1 - dones)
        self.critic_local.model.train_on_batch(x=[states, actions], y=Q_targets)

        def save_grads(writer, model):
            for layer in model.layers:
                for weight in layer.weights:
                    mapped_weight_name = weight.name.replace(':', '_')
                    tf.summary.histogram(mapped_weight_name, weight)

                    grads = model.optimizer.get_gradients(model.total_loss, weight)
                    def is_indexed_slices(grad):
                        return type(grad).__name__ == 'IndexedSlices'
                    grads = [grad.values if is_indexed_slices(grad) else grad for grad in grads]
                    tf.summary.histogram('{}_grad'.format(mapped_weight_name), grads)
                    merged = tf.summary.merge_all()
                    writer.flush()
                    writer.close()

        #save_grads(self.writer, self.critic_local.model)
        #def write_log(callback, names, logs, batch_no):
        #    for name, value in zip(names, logs):
        #        summary = tf.Summary()
        #        summary_value = summary.value.add()
        #        summary_value.simple_value = value
        #        summary_value.tag = name
        #        callback.writer.add_summary(summary, batch_no)
        #        callback.writer.flush()

        #train_names = ['train_loss', 'train_mae']
        #print("about to write log")
        #write_log(self.callback, train_names, logs, self.learn_counter)
        #trainable_weights = critic_local.model.trainable_weights
        #gradients = critic_local.model.optimizer.get_gradients(critic_local.model.total_loss, trainable_weights)

        # Train actor model (local)
        action_gradients = np.reshape(self.critic_local.get_action_gradients([states, actions, 0]), (-1, self.action_size))
        self.actor_local.train_fn([states, action_gradients, 1])  # custom training function

        # Soft-update target models
        self.soft_update(self.critic_local.model, self.critic_target.model)
        self.soft_update(self.actor_local.model, self.actor_target.model)

        #self.learn_counter += 1

    def soft_update(self, local_model, target_model):
        """Soft update model parameters."""
        local_weights = np.array(local_model.get_weights())
        target_weights = np.array(target_model.get_weights())

        assert len(local_weights) == len(target_weights), "Local and target model parameters must have the same size"

        new_weights = self.tau * local_weights + (1 - self.tau) * target_weights
        target_model.set_weights(new_weights)

    def _save_weight(self, model, directory_name, file_name):
        cwd = os.getcwd()
        directory_path = os.path.join(cwd, directory_name)
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)

        file_path = os.path.join(directory_path, file_name)

        mv_file_to_dir_with_date(file_path, directory_path)

        model.save_weights(file_path)

    def save_weights(self, location='weights_backup'):
        if self.verbose:
            print("start save_weights")

        self._save_weight(self.critic_local.model, location, "critic_local.h5")
        self._save_weight(self.critic_target.model, location, "critic_target.h5")
        self._save_weight(self.actor_local.model, location, "actor_local.h5")
        self._save_weight(self.actor_target.model, location, "actor_target.h5")

        if self.verbose:
            print("done save_weights")

    def _h5(self, model, file_path):
        if os.path.exists(file_path):
            model.load_weights(file_path)
        else:
            print(f'could not find weight to load from [{file_path}]')

    def load_weights(self, location='weights_backup'):
        if self.verbose:
            print("start load_weights")

        cwd = os.getcwd()
        directory_path = os.path.join(cwd, location)

        self._h5(self.critic_local.model, os.path.join(directory_path, "critic_local.h5"))
        self._h5(self.critic_target.model, os.path.join(directory_path, "critic_target.h5"))
        self._h5(self.actor_local.model, os.path.join(directory_path, "actor_local.h5"))
        self._h5(self.actor_target.model, os.path.join(directory_path, "actor_target.h5"))

        if self.verbose:
            print("done load_weights")

