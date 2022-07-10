import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gym
import math
from itertools import count
from .model import DQN_MODEL
from .memory import Memory
from oneagent import config


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


class DQN_Agent(object):
    def __init__(self, state_dim, action_dim, model_class, batch_size = BATCH_SIZE, lr = LR, gamma = GAMMA, target_policy_updated_step = TARGET_REPLACE_ITER, memory_buffer_size = MEMORY_CAPACITY, eps_start = EPS_START, eps_min = EPS_END, eps_decay = EPS_DECAY, build_traget_model = True, worker_id=0, tau=1, check_point_path=None, policy_name="DQN", loading_model=True):
        self.device = torch.device("cuda:%s"%worker_id if torch.cuda.is_available() else "cpu")
        self.name = policy_name
        self.check_point_path = check_point_path+self.name+".pth"
        self.loading_model = loading_model
        self.tau = tau
        self.rank = worker_id
        # Hyper Parameters
        self.input_shape = state_dim
        self.num_actions = action_dim
        self.batch_size = batch_size
        self.lr = lr
        self.gamma = gamma
        self.build_traget_model = build_traget_model
        self.target_policy_updated_step = target_policy_updated_step
        self.memory_buffer_size = memory_buffer_size
        self.eps_start = eps_start
        self.eps_min = eps_min
        self.eps_decay = eps_decay
        self.learn_step = 0
        self.update = False
        self.model_class = model_class
        self.initialize_model()
    
    def initialize_model(self):
        # initialize network  
        self.model = self.model_class(self.input_shape, self.num_actions)
        # loading model from checkpoint
        if self.loading_model:
            try:
                self.model.load_state_dict(torch.load(self.check_point_path))
            except EOFError as e:
                print("model is empty")
            except FileNotFoundError as e:
                print("cant find model")
        if self.build_traget_model:
            self.target_model = self.model_class(self.input_shape, self.num_actions)
            # synchronize 2 model
            self.target_model.load_state_dict(self.model.state_dict())
            self.target_model = self.target_model.to(self.device)
        self.model = self.model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.loss_func = nn.MSELoss()
        self.learn_step_counter = 0
        self.steps_done = 0
        # replay buffer
        self.memory = Memory(self.device, self.memory_buffer_size)
    
    def choose_action(self, x, ENV_A_SHAPE):
        x = torch.FloatTensor(x).to(self.device)
        eps_threshold = self.eps_min + (self.eps_start - self.eps_min) * math.exp(-1. * self.steps_done / self.eps_decay)
        self.steps_done += 1   
        if np.random.uniform() > eps_threshold: 
            actions_value = self.model.forward(x)
            action = torch.max(actions_value, 1)[1].data.cpu().numpy()
            action = action if ENV_A_SHAPE == 0 else action.reshape(ENV_A_SHAPE)
        else: 
            action = np.random.randint(0, self.num_actions, size=x.shape[0])
            action = action if ENV_A_SHAPE == 0 else action.reshape(ENV_A_SHAPE)
        return action
    
    def store_experience(self, experience):
        self.memory.add(reward=experience.reward, state=experience.state, done=experience.done, next_state=experience.next_state, action=experience.action)
    
    def soft_updated(self):
        model_weight = self.model.state_dict()
        target_model_weight = self.target_model.state_dict()
        for layer in target_model_weight.keys():
            target_model_weight[layer] = self.tau*model_weight[layer] + (1-self.tau)*target_model_weight[layer]
        self.target_model.load_state_dict(target_model_weight)

    def train(self):
        # if the buffer is not full, not train
        if self.memory.memory_counter <  self.memory_buffer_size:
            return None
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        q_eval = self.model(states).gather(1, actions)
        q_next = self.target_model(next_states).max(1)[0].view(self.batch_size, 1).detach()
        q_next = torch.mul(q_next, dones)
        q_target = rewards + self.gamma * q_next
        loss = self.loss_func(q_eval, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss
        #client.log_metric(key='loss', value=loss.item())
    
    def save_model(self):
        model_dict = self.model.state_dict()
        for key in model_dict: model_dict[key] = model_dict[key].clone().cpu()
        torch.save(model_dict, self.check_point_path)
