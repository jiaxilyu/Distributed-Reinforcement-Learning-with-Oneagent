import torch

class Memory:
    def __init__(self):
        self.rewards = []
        self.states = []
        self.dones = []
        self.last_state = None
        self.old_logprob = []
        self.actions = []
        self.episode = None
        
    def add(self, reward, state, done, action, old_logprob):
        self.rewards.append(reward)
        self.states.append(state)
        self.dones.append(done)
        self.actions.append(action)
        self.old_logprob.append(old_logprob)
        
    def clear(self):
        del self.rewards[:]
        del self.states[:]
        del self.dones[:]
        del self.actions[:]
        del self.old_logprob[:]
    
    def to_tensor(self):
        #self.rewards = [torch.tensor(reward) for reward in self.rewards]
        self.states = [torch.tensor(state) for state in self.states]
        self.actions = [torch.tensor(action) for action in self.actions]
        self.old_logprob = [torch.tensor(old_logprob.clone().detach()) for old_logprob in self.old_logprob]