import gym
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
import numpy as np
from .memory import Memory
from .model import Actor_network, Critic_network
from oneagent import config
import os

CRITIC_MODEL_PATH = config.CRITIC_MODEL_PATH
ACTOR_MODEL_PATH = config.ACTOR_MODEL_PATH

class DDPG_Agent:
    def __init__(self, state_dim, action_dim, action_low_bound, action_high_bound,
                 gamma = 0.9, lr = 0.001, buffer_size = 10000, batch_size = 32,
                 tau = 0.02, use_critic_network = True,
                 use_actor_network = True, use_target = True, target_update = 10, worker_id=0):
        self.device = torch.device("cuda:%s"%worker_id if torch.cuda.is_available() else "cpu")
        self.save_path = {"actor":ACTOR_MODEL_PATH, "critic":CRITIC_MODEL_PATH}
        self.use_target = use_target
        self.use_actor_network = use_actor_network
        self.use_critic_network = use_critic_network
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.lr = lr
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.target_update = target_update
        self.tau = tau
        self.critic_loss_func = nn.MSELoss()
        self.mu = 0.05
        self.min_mu = 0.0001
        self.mu_decays = 0.000001
        # soft update hyperparameter
        self.train_cnt = 0
        self.action_low_bound = action_low_bound
        self.action_high_bound = action_high_bound
        self.train_models = {}
        self.optim = optim.Adam
        self.initialize_model()

    def initialize_model(self):
        if self.use_actor_network:
            self.actor_policy = Actor_network(self.state_dim, self.action_dim, self.action_high_bound, self.action_low_bound)
            try:
                self.actor_policy.load_state_dict(torch.load(ACTOR_MODEL_PATH))
            except FileNotFoundError as e:
                path = "model"
                if not os.path.exists(path):
                    os.makedirs("model")
                print("cant find the model")
            except EOFError as e:
                print("model is empty")
            self.actor_policy = self.actor_policy.to(self.device)
            self.actor_optimizer = self.optim(self.actor_policy.parameters(), lr=self.lr)
            self.train_models["actor"] = self.actor_policy
            if self.use_target:
               self.actor_policy_target = Actor_network(self.state_dim, self.action_dim, self.action_high_bound, self.action_low_bound)
               self.actor_policy_target.load_state_dict(self.actor_policy.state_dict())
               self.actor_policy_target = self.actor_policy_target.to(self.device)

        if self.use_critic_network:
            self.critic_policy = Critic_network(self.state_dim, self.action_dim)
            try:
                self.critic_policy.load_state_dict(torch.load(CRITIC_MODEL_PATH))
            except FileNotFoundError as e:
                path = "model"
                if not os.path.exists(path):
                    os.makedirs("model")
                print("cant find the model")
            except EOFError as e:
                print("model is empty")
            self.critic_policy = self.critic_policy.to(self.device)
            self.critic_optimizer = self.optim(self.critic_policy.parameters(), lr=self.lr*2)
            self.train_models["critic"] = self.critic_policy
            # replay buffer
            self.memory = Memory(self.device, self.buffer_size)
            if self.use_target:
               self.critic_policy_target = Critic_network(self.state_dim, self.action_dim)
               self.critic_policy_target.load_state_dict(self.critic_policy.state_dict())
               self.critic_policy_target = self.critic_policy_target.to(self.device)
    
    def choose_action(self, state, use_noise = True):
        state = torch.FloatTensor(state).to(self.device)
        action = self.actor_policy(state).data.cpu().numpy()
        # for exploration
        if use_noise:
            self.mu = max(self.mu - self.train_cnt*self.mu_decays, self.min_mu)
            noise = np.random.normal(0, self.mu, 1)
            # make sure action within range
            action = np.clip(action + noise, self.action_low_bound, self.action_high_bound)
        return action

    def store_experience(self, experience):
        self.memory.add(reward=experience.reward, state=experience.state, done=experience.done, next_state=experience.next_state, action=experience.action)
    
    def train(self):
        # not start training yet
        if self.memory.memory_counter < self.buffer_size:
            return None, None
        self.train_cnt += 1
        # start training
        samples = self.memory.sample(self.batch_size)
        critic_loss = self.train_critic(samples)
        actor_loss = self.train_actor(samples)
        #soft update actor network and critic network
        if self.train_cnt % self.target_update == 0:
            self.soft_update(self.actor_policy, self.actor_policy_target)
            self.soft_update(self.critic_policy, self.critic_policy_target)
        return critic_loss, actor_loss
        
    
    def train_critic(self, samples):
        states, actions, rewards, next_states, dones = samples
        # train critic
        
        # step1 using target actor network for choosing the next_states action
        next_states_actions = self.actor_policy_target(next_states)

        actions = actions.squeeze(1)
        next_states_actions = next_states_actions.squeeze(0).detach()
        # step2 get Q(s,a) by critic
        q = self.critic_policy(states, actions)

        # step3 get target Q(s,a) = reward + Q'(s',a')*done*gamma
        q_next = self.critic_policy_target(next_states, next_states_actions).detach()
        q_next = torch.mul(q_next, dones)
        q_target = rewards + self.gamma * q_next
        
        # step4 loss backward
        critic_loss =  self.critic_loss_func(q, q_target)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        return critic_loss
    
    def train_actor(self, samples):
        states, actions, rewards, next_states, dones = samples
        # train actor_network
        # cannot reuse action value, since the old action value are calculated by old network
        actions = self.actor_policy(states)
        q = self.critic_policy(states, actions)
        actor_loss = -torch.mean(q)
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        return actor_loss

    def soft_update(self, model, target_model):
        for param, target_param in zip(model.parameters(), target_model.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)
    
    def save_model(self):
        if self.use_actor_network:
            model_dict = self.actor_policy.state_dict()
            for key in model_dict: model_dict[key] = model_dict[key].clone().cpu()
            torch.save(model_dict, self.save_path["actor"])

        if self.use_critic_network:
            model_dict = self.critic_policy.state_dict()
            for key in model_dict: model_dict[key] = model_dict[key].clone().cpu()
            torch.save(model_dict, self.save_path["critic"])
    
    def get_agent_dict(self, model_name):
        model = self.train_models[model_name]
        return model.state_dict()
    