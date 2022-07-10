import gym
import torch
import torch.nn as nn
import numpy as np
from copy import deepcopy
import torch.optim as optim
import numpy as np
import time
from itertools import count
from collections import namedtuple


class Actor_network(nn.Module):
    def __init__(self, state_shape, num_actions, action_high_bound, action_low_bound):
        super(Actor_network, self).__init__()
        self.state_shape = state_shape
        self.num_actions = num_actions
        self.action_high_bound = float(action_high_bound)
        self.action_low_bound = float(action_low_bound)
        self.action_midpoint = (self.action_high_bound + self.action_low_bound)/2
        self.scalar = (self.action_high_bound - self.action_low_bound)/2
        # network structure
        self.fc1 = nn.Linear(state_shape, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, self.num_actions)
        
        # initialize fc3 weight
        self.initialize_weight()

    def initialize_weight(self):
        self.fc1.weight.data.normal_(0, 0.1)
        self.fc2.weight.data.normal_(0, 0.1)
        self.fc3.weight.data.normal_(0, 0.1)

    def forward(self, state):
        x = self.fc1(state)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        result = torch.tanh(x)*self.scalar + self.action_midpoint
        return result

class Critic_network(nn.Module):
    def __init__(self, state_shape, num_actions):
        super(Critic_network, self).__init__()
        self.state_shape = state_shape
        self.num_actions = num_actions

        # network structure
        self.fc1 = nn.Linear(state_shape + num_actions, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 1)

        # initialize fc3 weight
        self.initialize_weight()
    
    def initialize_weight(self):
        self.fc1.weight.data.normal_(0, 0.1)
        self.fc2.weight.data.normal_(0, 0.1)
        self.fc3.weight.data.normal_(0, 0.1)
    
    def forward(self, states, actions):
        x = self.fc1(torch.cat([states, actions], 1))
        x = self.relu(x)
        # then input action info
        x = self.fc2(x)
        x = self.relu(x)
        result = self.fc3(x)
        return result

def test_network():
    state_shape = 5
    action_dim = 5
    actor = Actor_network(state_shape, action_dim)
    critic = Critic_network(state_shape, action_dim)
    states = torch.randn(300, 5)
    actions = actor(states)
    print(critic(states, actions))