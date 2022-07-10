import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd

torch.manual_seed(543)
class DQN_MODEL(nn.Module):
    def __init__(self, N_STATES, N_ACTIONS):
        super(DQN_MODEL, self).__init__()
        self.fc1 = nn.Linear(N_STATES, 256)
        self.fc1.weight.data.normal_(0, 0.1) 
        self.out = nn.Linear(256, N_ACTIONS)
        self.out.weight.data.normal_(0, 0.1) 

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        actions_value = self.out(x)
        return actions_value

class DuelingDQNModel(nn.Module):
    def __init__(self, N_STATES, N_ACTIONS):
        super(DuelingDQNModel, self).__init__()
        self.fc1 = nn.Linear(N_STATES, 256)
        self.fc1.weight.data.normal_(0, 0.1) 
        self.advantages = nn.Linear(256, N_ACTIONS)
        self.advantages.weight.data.normal_(0, 0.1)
        self.v_out = nn.Linear(256, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        advantages = self.advantages(x)
        advantages = advantages - advantages.mean().detach()
        v_out = self.v_out(x)
        return v_out + advantages

