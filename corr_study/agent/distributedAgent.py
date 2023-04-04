import torch
import copy
import numpy as np
from torch.nn import Module
from agent.agent import Agent, UEAgent
from agent.q_learning import DQL
from agent.neural_network import LinearNeuralNetwork
from plot.linear import linear_plot, multi_linear_plot
import seaborn as sns
from abc import ABC, abstractmethod

class DistributedAgent(UEAgent):
    """Class representing an UE Agent in distributed approach.
    This means that the Agent will perform operations only on the data
    directly related to this user
    """

    def __init__(self, 
                 imsi: int, 
                 step_num: int, 
                 episode_num: int, 
                 state_dim: int, 
                 action_num: int, 
                 user_num: int, 
                 state_labels: [str], 
                 action_labels: [str], 
                 state_normalization: [[]], 
                 state_mask: np.ndarray,
                 gamma: float,
                 batch_size: int,
                 target_replace: int,
                 memory_capacity: int,
                 learning_rate: float,
                 eps: float,
                 weight_decay: float,
                 format=None,
                 standard=False):
        super().__init__(imsi, 
                         step_num, 
                         episode_num, 
                         state_dim, 
                         action_num, 
                         user_num, 
                         state_labels, 
                         action_labels, 
                         state_normalization, 
                         state_mask,
                         format)

        # Primary and target networks used in the training
        # The primary network is used to choose new actions
        # The target network is used to predict the future q values

        self.primary_net: Module = LinearNeuralNetwork(self.learning_state_dim, self.action_num)
        target_net: Module = LinearNeuralNetwork(self.learning_state_dim, self.action_num)
        target_net.load_state_dict(self.primary_net.state_dict())

        if standard:
            params = torch.load('standard/' + 'model')
            self.primary_net.load_state_dict(params)
            target_net.load_state_dict(params)

        # Learning weights optimizer

        optimizer = torch.optim.Adam(self.primary_net.parameters(),
                                     lr=learning_rate, eps=eps, weight_decay=weight_decay)

        # Double Q Learning Algorithm

        self.dql = DQL(self.primary_net,
                       target_net,
                       optimizer,
                       self.action_num,
                       gamma,
                       batch_size,
                       target_replace,
                       memory_capacity)

        # Initialize learning transition (state, action, reward, new_state)

        self.imsi_list = []
        self.states = []
        self.action_indexes = []
        self.rewards = []
        self.old_imsi_list = []
        self.old_states = []
        self.old_action_indexes = []

        self.data_idx = -1

        # Learning data

        self.state_data = np.zeros((self.state_dim, step_num * episode_num), dtype=np.float32)
        self.action_data = np.zeros((self.action_num, step_num * episode_num), dtype=np.float32)
        self.q_value_data = np.zeros((self.action_num, step_num * episode_num), dtype=np.float32)

        self.chamfer_data = np.zeros((step_num * episode_num), dtype=np.float32)
        self.qos_data = np.zeros((step_num * episode_num), dtype=np.float32)
        self.reward_data = np.zeros((step_num * episode_num), dtype=np.float32)
        self.temperature_data = np.zeros(step_num * episode_num, dtype=np.float32)
        self.loss_data = np.zeros(step_num * episode_num, dtype=np.float32)

    def reset(self):
        self.imsi_list = []
        self.states = []
        self.action_indexes = []
        self.rewards = []
        self.old_imsi_list = []
        self.old_states = []
        self.old_action_indexes = []

    def get_action(self, 
                   imsi_list: [int],
                   states: [np.ndarray], 
                   temp: float):

        # assert 0 <= temp <= 1
        
        action_indexes, q_values = [], []

        state = states[0]

        if state is None:

            user_action_idx = None
            user_q_values = None

        else:

            x = torch.tensor(state[self.state_mask], dtype=torch.float32)

            # Estimate the q values of the input state

            with torch.no_grad():
                user_q_values = self.primary_net.forward(x).detach().numpy()

            # Choose an action according to the epsilon greedy policy

            if np.random.uniform() - temp > 0:
                user_action_idx = np.argmax(user_q_values, 0)  # Greedy action
            else:
                user_action_idx = np.random.randint(0, self.action_num)  # Random action

        action_indexes.append(user_action_idx)
        q_values.append(user_q_values)

        return action_indexes, q_values
    
    def update(self,
               imsi_list: [int],
               action_indexes: [int],
               q_values: [np.ndarray],
               rewards: [float],
               states: [np.ndarray],
               qos_per_user: [float],
               cd_per_user: [float],
               temp: float,
               train: bool):

        self.data_idx += 1

        
        self.temperature_data[self.data_idx] = temp
        
        # we get in input lists but the values corresponding to this agent are always at index 0
        # user_idx = 0

        try:
            imsi_idx = 0
            action_idx = action_indexes[imsi_idx]
            assert action_idx is not None
            action_vector = np.zeros(self.action_num)
            action_vector[action_idx] = 1
            self.action_data[:, self.data_idx] = action_vector
            self.q_value_data[:, self.data_idx] = q_values[imsi_idx].T

            self.reward_data[self.data_idx] = rewards[imsi_idx]

            
            self.state_data[:, self.data_idx] = states[imsi_idx]
            
            self.qos_data[self.data_idx] = qos_per_user[imsi_idx]

            self.chamfer_data[self.data_idx] = cd_per_user[imsi_idx]

        except:

            self.action_data[:, self.data_idx] = self.action_data[:, self.data_idx - 1]
            self.q_value_data[:, self.data_idx] = self.q_value_data[:, self.data_idx -1]
            self.reward_data[self.data_idx] = self.reward_data[self.data_idx - 1]
            self.state_data[:, self.data_idx] = self.state_data[:, self.data_idx - 1]
            self.qos_data[self.data_idx] = self.qos_data[self.data_idx - 1] 
            self.chamfer_data[self.data_idx] = self.chamfer_data[self.data_idx - 1]

        
        # Update the transition variables

        self.old_imsi_list = copy.deepcopy(self.imsi_list)
        self.old_states = copy.deepcopy(self.states)
        self.old_action_indexes = copy.deepcopy(self.action_indexes)

        self.imsi_list = []
        self.action_indexes = []
        self.rewards = []
        self.states = []

        for imsi, action_idx, reward, state in zip(imsi_list, action_indexes, rewards, states):

            if action_idx is not None:

                self.imsi_list.append(imsi)
                self.action_indexes.append(action_idx)
                self.rewards.append(reward)
                self.states.append(state)

        if train:

            # If the state variable is not None, store the new transition in the replay memory

            for old_idx, old_imsi in enumerate(self.old_imsi_list):
                if old_imsi in self.imsi_list:
                    new_idx = self.imsi_list.index(old_imsi)
                    self.dql.store_transition(np.copy(self.old_states[old_idx]),
                                              self.old_action_indexes[old_idx],
                                              self.rewards[new_idx],
                                              np.copy(self.states[new_idx]))

            # If the replay memory is full, perform a learning step

            if self.dql.ready():

                self.loss_data[self.data_idx] = self.dql.step()

    #TODO came up with a clever idea on how to manage data_folder
    def save_data(self, data_folder: str):

        """
        Save the learning data
        """
        np.save(data_folder + 'states.npy', self.state_data)
        np.save(data_folder + 'rewards.npy', self.reward_data)
        np.save(data_folder + 'actions.npy', self.action_data)
        np.save(data_folder + 'q_values.npy', self.q_value_data)
        np.save(data_folder + 'temperatures.npy', self.temperature_data)
        np.save(data_folder + 'losses.npy', self.loss_data)
        np.save(data_folder + 'qos.npy', self.qos_data)
        np.save(data_folder + 'chamfer_distances.npy', self.chamfer_data)
        np.save(data_folder + 'data_idx.npy', self.data_idx)

    def save_model(self, data_folder: str):

        self.dql.save_model(data_folder)

    def load_data(self, data_folder: str):

        """
        Load the learning data
        """

        self.data_idx = np.load(data_folder + 'data_idx.npy')
        self.state_data = np.load(data_folder + 'states.npy')
        self.reward_data = np.load(data_folder + 'rewards.npy')
        self.action_data = np.load(data_folder + 'actions.npy')
        self.q_value_data = np.load(data_folder + 'q_values.npy')
        self.temperature_data = np.load(data_folder + 'temperatures.npy')
        self.loss_data = np.load(data_folder + 'losses.npy')
        self.qos_data = np.load(data_folder + 'qos.npy')
        self.chamfer_data = np.load(data_folder + 'chamfer_distances.npy')

    def load_model(self, data_folder: str):

        self.dql.load_model(data_folder)

    def plot_data(self, data_folder: str, episode_num: int):

        """
        Plot the learning data
        """
        pass

class MultiDistributedAgent(Agent):
    def __init__(self, 
                 step_num: int, 
                 episode_num: int, 
                 state_dim: int, 
                 action_num: int, 
                 user_num: int, 
                 state_labels: [str], 
                 action_labels: [str], 
                 state_normalization: [[]], 
                 state_mask: np.ndarray,
                 gamma: float,
                 batch_size: int,
                 target_replace: int,
                 memory_capacity: int,
                 learning_rate: float,
                 eps: float,
                 weight_decay: float,
                 format=None,
                 standard=False):


        super().__init__(step_num, 
                         episode_num, 
                         state_dim, 
                         action_num, 
                         user_num, 
                         state_labels, 
                         action_labels, 
                         state_normalization, 
                         state_mask,
                         format)

        self.users = {}
        self.users_imsi = []
        for imsi in range(1, user_num + 1):
            self.users[imsi] = DistributedAgent(imsi, 
                                                step_num, 
                                                episode_num, 
                                                state_dim, 
                                                action_num, 
                                                user_num, 
                                                state_labels, 
                                                action_labels, 
                                                state_normalization, 
                                                state_mask,
                                                gamma,
                                                batch_size,
                                                target_replace,
                                                memory_capacity,
                                                learning_rate,
                                                eps,
                                                weight_decay,
                                                format, 
                                                standard)
            self.users_imsi += [imsi,]

    def reset(self):
        for imsi in self.users_imsi:
            self.users[imsi].reset()

    def get_action(self, 
                   imsi_list: [int],
                   states: [np.ndarray], 
                   temp: float):

        imsi = imsi_list[0]
        return self.users[imsi].get_action(imsi_list, states, temp)
    
    def update(self,
               imsi_list: [int],
               action_indexes: [int],
               q_values: [np.ndarray],
               rewards: [float],
               states: [np.ndarray],
               qos_per_user: [float],
               cd_per_user: [float],
               temp: float,
               train: bool):

        imsi = imsi_list[0]
        self.users[imsi].update(imsi_list, 
                                action_indexes, 
                                q_values,
                                rewards,
                                states,
                                qos_per_user,
                                cd_per_user,
                                temp,
                                train)

    def save_data(self, data_folder: str):
        for imsi in self.users_imsi:
            self.users[imsi].save_data(data_folder + '/imsi=' + str(imsi) + "/")

    def save_model(self, data_folder: str):
        for imsi in self.users_imsi:
            self.users[imsi].save_model(data_folder + '/imsi=' + str(imsi) + "/")

    def load_data(self, data_folder: str):
        for imsi in self.users_imsi:
            self.users[imsi].load_data(data_folder + '/imsi=' + str(imsi) + "/")

    def load_model(self, data_folder: str):
        for imsi in self.users_imsi:
            self.users[imsi].load_model(data_folder + '/imsi=' + str(imsi) + "/")

    def plot_data(self, data_folder: str, episode_num: int):

        state_palette = sns.color_palette('rocket', n_colors=self.learning_state_dim)
        action_palette = sns.color_palette('rocket_r', n_colors=3)
        user_palette = sns.color_palette('rocket_r', n_colors=self.user_num)
        single_palette = sns.color_palette('rocket', n_colors=1)

        multi_data = []
        multi_keys = []
        for imsi in self.users_imsi:
            multi_data += [self.users[imsi].q_value_data[i, :self.users[imsi].data_idx] for i in range(self.action_num)]
            multi_keys += [str(imsi) + " " + str(a) for a in self.action_labels]

        multi_linear_plot(multi_data,
                        multi_keys,
                        'Episode',
                        'Q value',
                        episode_num,
                        data_folder + '/learn/q_values',
                        palette= sns.color_palette('rocket_r', n_colors=len(multi_keys)),
                        plot_format=self.format)
        # for user_idx in range(self.user_num):
        for imsi in self.users_imsi:

            ### LEARN PLOT ###

            user_folder = data_folder + 'learn/' + str(imsi-1) + '/'

            # States

            multi_data = [self.users[imsi].state_data[i, :self.users[imsi].data_idx] for i in range(self.state_dim) if
                          self.state_mask[i]]

            multi_keys = np.array(self.state_labels)[self.state_mask]

            multi_linear_plot(multi_data,
                              multi_keys,
                              'Episode',
                              'State',
                              episode_num,
                              user_folder + 'states',
                              palette=state_palette,
                              plot_format=self.format)

            # Q values

            multi_data = [self.users[imsi].q_value_data[i, :self.users[imsi].data_idx] for i in range(self.action_num)]
            multi_keys = self.action_labels

            multi_linear_plot(multi_data,
                              multi_keys,
                              'Episode',
                              'Q value',
                              episode_num,
                              user_folder + 'q_values',
                              palette=action_palette,
                              plot_format=self.format)

            # Actions

            multi_data = [self.users[imsi].action_data[i, :self.users[imsi].data_idx] for i in range(self.action_num)]
            multi_keys = self.action_labels

            multi_linear_plot(multi_data,
                              multi_keys,
                              'Episode',
                              'Action probability',
                              episode_num,
                              user_folder + 'actions',
                              palette=action_palette,
                              plot_format=self.format)

            # Reward

            linear_plot(self.users[imsi].reward_data[:self.users[imsi].data_idx],
                        'Episode',
                        'Reward',
                        episode_num,
                        user_folder + 'rewards',
                        palette=single_palette,
                        plot_format=self.format)

            ### SINGLE FEATURE PLOT ###

            user_folder = data_folder + 'state/' + str(imsi-1) + '/'

            for i in range(self.state_dim):
                min_value, max_value = self.state_normalization[i]

                linear_plot(self.users[imsi].state_data[i, :self.users[imsi].data_idx] * (max_value - min_value) + min_value,
                            'Episode',
                            self.state_labels[i],
                            episode_num,
                            user_folder + self.state_labels[i].replace(' ', '_'),
                            palette=single_palette,
                            plot_format=self.format)

            ### PERFORMANCE PLOT ###

            user_folder = data_folder + 'performance/' + str(imsi-1) + '/'

            linear_plot(self.users[imsi].chamfer_data[:self.users[imsi].data_idx],
                        'Episode',
                        'Chamfer Distance',
                        episode_num,
                        user_folder + 'chamfer_distances',
                        palette=single_palette,
                        plot_format=self.format)

            linear_plot(
                (self.max_penalty - self.users[imsi].chamfer_data[:self.users[imsi].data_idx]) / self.max_penalty,
                'Episode',
                'QoE',
                episode_num,
                user_folder + 'qoe',
                palette=single_palette,
                plot_format=self.format)
        
        multi_chamfer_data = [self.users[imsi].chamfer_data[:self.users[imsi].data_idx] 
                              for imsi in self.users_imsi]
        multi_chamfer_label = ["imsi = " + str(imsi) for imsi in self.users_imsi]

        multi_linear_plot(multi_chamfer_data,
                          multi_chamfer_label,
                          'Episode',
                          'Chamfer Distance',
                          episode_num,
                          data_folder + 'performance/chamfer_distances',
                          palette=user_palette,
                          plot_format=self.format)


        multi_qoe_data = [(self.max_penalty - self.users[imsi].chamfer_data[:self.users[imsi].data_idx]) / self.max_penalty
                          for imsi in self.users_imsi] 
        multi_linear_plot(multi_qoe_data,
                          multi_chamfer_label,
                          'Episode',
                          'QoE',
                          episode_num,
                          data_folder + 'performance/qoe',
                          palette=user_palette,
                          plot_format=self.format)
