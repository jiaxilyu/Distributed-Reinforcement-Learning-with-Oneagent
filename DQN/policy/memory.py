import torch
import numpy as np

class Memory:
    def __init__(self, device, MEMORY_CAPACITY = 10000):
        self.device = device
        self.memory_capacity =  MEMORY_CAPACITY
        self.rewards = [None for i in range(MEMORY_CAPACITY)]
        self.states =  [None for i in range(MEMORY_CAPACITY)]
        self.dones =  [None for i in range(MEMORY_CAPACITY)]
        self.next_states =  [None for i in range(MEMORY_CAPACITY)]
        self.actions =  [None for i in range(MEMORY_CAPACITY)]
        self.memory_counter = 0
        
    def add(self, reward, state, done, next_state, action):
        index = self.memory_counter % self.memory_capacity
        self.rewards[index] = torch.FloatTensor([reward]).to(self.device)
        self.states[index] = torch.FloatTensor(state).to(self.device)
        self.dones[index] = torch.FloatTensor([done]).to(self.device)
        self.next_states[index] = torch.FloatTensor(next_state).to(self.device)
        self.actions[index] = torch.tensor([action]).to(self.device)
        self.memory_counter += 1
    
    def sample(self, batch_size):
        sample_index = np.random.choice(self.memory_capacity, batch_size)
        sample_index = [int(index) for index in sample_index]
        states = torch.stack([self.states[index] for index in sample_index])
        actions = torch.stack([self.actions[index] for index in sample_index])
        rewards = torch.stack([self.rewards[index] for index in sample_index])
        next_states = torch.stack([self.next_states[index] for index in sample_index])
        dones = torch.stack([self.dones[index] for index in sample_index])
        return states, actions, rewards, next_states, dones