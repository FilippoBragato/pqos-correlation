import torch
import copy
import numpy as np
from torch.nn import Module
from .q_learning import DQL
from .neural_network import LinearNeuralNetwork

class Agent():

    def __init__(self,
                 step_num: int,
                 episode_num: int,
                 state_dim: int,
                 action_num: int,
                 state_labels: [str],
                 action_labels: [str],
                 state_normalization: [[]],
                 learning_rate,
                 eps,
                 weight_decay,
                 gamma,
                 batch_size,
                 target_replace,
                 memory_capacity):
        # Number of steps and episodes

        self.step_num, self.episode_num = step_num, episode_num

        # State dimension

        self.state_dim: int = state_dim

        # Number of possible actions in each state

        self.action_num: int = action_num


        # State and action labels

        self.state_labels = state_labels
        self.action_labels = action_labels
        self.state_normalization = state_normalization
        self.max_penalty = None

        self.primary_net: Module = LinearNeuralNetwork(state_dim, self.action_num)
        target_net: Module = LinearNeuralNetwork(state_dim, self.action_num)
        target_net.load_state_dict(self.primary_net.state_dict())

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

        self.state = []
        self.action_index = []
        self.reward = []
        self.old_state = []
        self.old_action_index = []

        self.data_idx = -1

        # Learning data

        self.state_data = np.zeros((self.state_dim, step_num * episode_num), dtype=np.float32)
        self.action_data = np.zeros((self.action_num, step_num * episode_num), dtype=np.float32)
        self.q_value_data = np.zeros((self.action_num, step_num * episode_num), dtype=np.float32)

        self.reward_data = np.zeros((step_num * episode_num), dtype=np.float32)
        self.temperature_data = np.zeros(step_num * episode_num, dtype=np.float32)
        self.loss_data = np.zeros(step_num * episode_num, dtype=np.float32)
        
    
    def reset(self):
        """Delete all temporary stored data
        """
        self.state = []
        self.action_index = []
        self.reward = []
        self.old_state = []
        self.old_action_index = []

    def get_action(self,
                   state: np.ndarray,
                   temp: float):
        

        x = torch.tensor(state, dtype=torch.float32)

        # Estimate the q values of the input state

        with torch.no_grad():
            user_q_values = self.primary_net.forward(x).detach().numpy()

        # Choose an action according to the epsilon greedy policy

        if np.random.uniform() - temp > 0:
            user_action_idx = np.argmax(user_q_values, 0)  # Greedy action
        else:
            user_action_idx = np.random.randint(0, self.action_num)  # Random action


        return user_action_idx, user_q_values

    def update(self,
               action_index: int,
               q_values: np.ndarray,
               reward: float,
               state: np.ndarray,
               temp: float,
               train: bool):
        """Update the learning data of the agent

        Args:
            imsi_list ([int]): list of identifiers of the users
            action_indexes ([int]): list of action choosen by each user
            q_values ([np.ndarray]): Q-values obtained by the users
            rewards ([float]): rewards obtained by each user
            states ([np.ndarray]): the State array for each user
            qos_per_user ([float]): the Quality of Service obtained by each user
            cd_per_user ([float]): the Chamfender Distance obtained by each user 
            temp (float): the Temperature of the episode
            train (bool): true if the algorithm is training
        """
        self.data_idx += 1

        
        # self.temperature_data[self.data_idx] = temp
        
        # we get in input lists but the values corresponding to this agent are always at index 0
        # user_idx = 0

        action_idx = action_index
        action_vector = np.zeros(self.action_num)
        action_vector[action_idx] = 1
        self.action_data[:, self.data_idx] = action_vector
        self.q_value_data[:, self.data_idx] = q_values.T #TODO check this T

        self.reward_data[self.data_idx] = reward
        
        self.state_data[:, self.data_idx] = state

        
        # Update the transition variables
        self.old_state = copy.deepcopy(self.state)
        self.old_action_index = copy.deepcopy(self.action_index)

        self.action_index = action_idx
        self.reward = reward
        self.state = state

        if train:

            # If the state variable is not None, store the new transition in the replay memory

            self.dql.store_transition(np.copy(self.old_state),
                                        self.old_action_index,
                                        self.reward,
                                        np.copy(self.state))

            # If the replay memory is full, perform a learning step

            if self.dql.ready():
                self.loss_data[self.data_idx] = self.dql.step()


    def save_data(self, data_folder: str):
        """Save the learning data

        Args:
            data_folder (str): position where to store learning data
        """
        np.save(data_folder + 'states.npy', self.state_data)
        np.save(data_folder + 'rewards.npy', self.reward_data)
        np.save(data_folder + 'actions.npy', self.action_data)
        np.save(data_folder + 'q_values.npy', self.q_value_data)
        np.save(data_folder + 'temperatures.npy', self.temperature_data)
        np.save(data_folder + 'losses.npy', self.loss_data)
        np.save(data_folder + 'data_idx.npy', self.data_idx)


    def save_model(self, data_folder: str):
        """Save the learning Neural Network

        Args:
            data_folder (str): location where to store the Neural Network
        """
        self.dql.save_model(data_folder)


    def load_data(self, data_folder: str):
        """Load learning data

        Args:
            data_folder (str): location where learning data are stored
        """

        self.data_idx = np.load(data_folder + 'data_idx.npy')
        self.state_data = np.load(data_folder + 'states.npy')
        self.reward_data = np.load(data_folder + 'rewards.npy')
        self.action_data = np.load(data_folder + 'actions.npy')
        self.q_value_data = np.load(data_folder + 'q_values.npy')
        self.temperature_data = np.load(data_folder + 'temperatures.npy')
        self.loss_data = np.load(data_folder + 'losses.npy')


    def load_model(self, data_folder: str):
        """Load the learning NN

        Args:
            data_folder (str): folder containing the Neural Network
        """

        self.dql.load_model(data_folder)
