import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gym
import math
from itertools import count
from .DQN_Agent import DQN_Agent
from .memory import Memory

# Hyper Parameters
# need to do : read from config.py
BATCH_SIZE = 32
LR = 0.001                   # learning rate
GAMMA = 0.99                 # reward discount
TARGET_REPLACE_ITER = 100   # target update frequency
MEMORY_CAPACITY = 10000
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 2000

class DDQN_Agent(DQN_Agent):
    def __init__(self, state_dim, action_dim, model_class, batch_size = BATCH_SIZE, lr = LR, gamma = GAMMA, target_policy_updated_step = TARGET_REPLACE_ITER, memory_buffer_size = MEMORY_CAPACITY, eps_start = EPS_START, eps_min = EPS_END, eps_decay = EPS_DECAY, build_traget_model = True, worker_id=0, check_point_path=None, policy_name=None, loading_model=True):
        super(DDQN_Agent, self).__init__(state_dim, action_dim, model_class, batch_size = BATCH_SIZE, lr = LR, gamma = GAMMA, target_policy_updated_step = TARGET_REPLACE_ITER, memory_buffer_size = MEMORY_CAPACITY, eps_start = EPS_START, eps_min = EPS_END, eps_decay = EPS_DECAY, build_traget_model = build_traget_model, worker_id=worker_id, check_point_path=check_point_path, policy_name=policy_name, loading_model=loading_model)
        self.name = policy_name
    
    def train(self):
        # if the buffer is not full, not train
        if self.memory.memory_counter <  self.memory_buffer_size:
            return None
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        q_eval = self.model(states).gather(1, actions)
        # use current network to choose next states action for target q, therefore Q(s_, arg_max(Q(s_, q_network)), q_target_network) <= max(Q(s_,_, q_target_network))
        # reduce the overestimation of q_next
        next_actions = torch.argmax(self.model(next_states), dim=1).view(actions.shape).detach()
        # use target network to calculate each pair Q(next_state, next_action)
        q_next = self.target_model(next_states).gather(1, next_actions).detach()
        q_next = torch.mul(q_next, dones)
        q_target= rewards + self.gamma*q_next
        loss = self.loss_func(q_eval, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss

