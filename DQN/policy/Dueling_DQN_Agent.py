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

class Dueling_DQN_Agent(DQN_Agent):
    def __init__(self, state_dim, action_dim, model_class, batch_size = BATCH_SIZE, lr = LR, gamma = GAMMA, target_policy_updated_step = TARGET_REPLACE_ITER, memory_buffer_size = MEMORY_CAPACITY, eps_start = EPS_START, eps_min = EPS_END, eps_decay = EPS_DECAY, build_traget_model = True, worker_id=0, check_point_path=None, policy_name=None, loading_model=True):
        super(Dueling_DQN_Agent, self).__init__(state_dim, action_dim, model_class, batch_size = BATCH_SIZE, lr = LR, gamma = GAMMA, target_policy_updated_step = TARGET_REPLACE_ITER, memory_buffer_size = MEMORY_CAPACITY, eps_start = EPS_START, eps_min = EPS_END, eps_decay = EPS_DECAY, build_traget_model = build_traget_model, worker_id=worker_id, check_point_path=check_point_path, policy_name=policy_name, loading_model=loading_model)
        self.name = policy_name