import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
from torch.distributions import Categorical
from torch.distributions import Normal


class Discrete_ActorModel(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Discrete_ActorModel, self).__init__()
        self.line1 = nn.Linear(state_dim, 128)
        self.dropout = nn.Dropout(p=0.6)
        self.line2 = nn.Linear(128, action_dim)
    
    def forward(self):
        raise NotImplementedError
        
    def actor(self, x):
        x = self.line1(x)
        x = self.dropout(x)
        x = F.relu(x)
        x = self.line2(x)
        return F.softmax(x, dim=-1)
    
    def evaluate(self, state, action, action_std = None):
        action_probs = self.actor(state)
        dist = Categorical(action_probs)
        action_logprobs = dist.log_prob(action)
        return action_logprobs, dist.entropy()


class Discrete_CriticModel(nn.Module):
    def __init__(self, state_dim):
        super(Discrete_CriticModel, self).__init__()
        self.line1 = nn.Linear(state_dim, 128)
        self.dropout = nn.Dropout(p=0.6)
        self.line2 = nn.Linear(128, 1)
        
    def forward(self, x):
        x = self.line1(x)
        x = self.dropout(x)
        x = F.relu(x)
        x = self.line2(x)
        return x

class Continous_ActorModel(nn.Module):  
    def __init__(self, state_dim, action_dim, max_action):
        super(Continous_ActorModel, self).__init__()
        self.max_action = max_action
        self.relu = nn.ReLU()
        self.hidden1 = nn.Linear(state_dim, 256)
        self.hidden2 = nn.Linear(256, 128)
        self.mu_layer = nn.Linear(128, action_dim)
        self.initialize_weight()
        
    def forward(self, x):
        x = torch.tanh(self.hidden1(x))
        x = torch.tanh(self.hidden2(x))
        mu = self.mu_layer(x)
        return mu
    
    # return the p(at|st)
    def evaluate(self, states, actions, action_std):
        mu = self.forward(states)
        std = action_std * torch.ones_like(mu)
        dist = Normal(mu, std)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        return log_probs, entropy
    
    def initialize_weight(self):
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
               nn.init.xavier_normal_(layer.weight)
        
class Continous_CriticModel(nn.Module):
    def __init__(self, state_dim):
        super(Continous_CriticModel, self).__init__()
        self.hidden1 = nn.Linear(state_dim, 256)
        self.hidden2 = nn.Linear(256, 128)
        self.output = nn.Linear(128, 1)
        
    def forward(self, state):
        x = F.relu(self.hidden1(state))
        x = F.relu(self.hidden2(x))
        value = self.output(x)
        return value
