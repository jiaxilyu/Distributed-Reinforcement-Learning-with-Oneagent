import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
from torch.distributions import Categorical
import gym
import numpy as np
import time
from .model import Discrete_ActorModel, Discrete_CriticModel, Continous_ActorModel, Continous_CriticModel
import os
from oneagent import config

TRAIN_EPOCH = 10
NBATCH = 1
CLIP_EPSILON = 0.2

CRITIC_MODEL_PATH = config.CRITIC_MODEL_PATH
ACTOR_MODEL_PATH = config.ACTOR_MODEL_PATH

# PPO version: stable baseline 3
class Base_PPO_Agent:
    def __init__(self, state_dim, action_dim, gamma = 0.99, lam = 0.95, critic_loss_coef = 0.5, 
                 entropy_coef = 0.00155, nopstepochs = 4, lr = 0.001, log_interval = 1, max_grad_norm = 0.75,
                 TRAIN_EPOCH = 10, NBATCH = 1, CLIP_EPSILON = 0.2,
                 target_kl = 5,use_critic_network = True, use_actor_network = True, device_id = 0):
        self.device = torch.device("cuda:%s"%device_id if torch.cuda.is_available() else "cpu")
        self.save_path = {"actor":ACTOR_MODEL_PATH, "critic":CRITIC_MODEL_PATH}
        self.use_actor_network = use_actor_network
        self.use_critic_network = use_critic_network
        self.max_grad_norm = max_grad_norm
        self.target_kl = target_kl
        self.eps = np.finfo(np.float64).eps.item() 
        self.gamma = gamma
        self.lam = lam
        self.lr = lr
        self.nopstepochs = nopstepochs
        self.log_interval = log_interval
        self.critic_loss_coef = critic_loss_coef
        self.entropy_coef = entropy_coef
        self.use_actor_network = use_actor_network
        self.use_critic_network = use_critic_network
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.TRAIN_EPOCH = TRAIN_EPOCH
        self.NBATCH = NBATCH
        self.CLIP_EPSILON = CLIP_EPSILON
        # this part is for ddp
        self.train_models = {}
        self.optim = optim.Adam
        # init policy network and critic network
    
    # reset model, optimzer, loss of agent for ddp mode adaptation
    def reset_ddp_adaptor(self, reset_model, optimizer):
        self.actor_policy = reset_model["actor"]
        self.actor_optimizer = optimizer["actor"]
        self.critic_policy = reset_model["critic"]
        self.critic_optimizer = optimizer["critic"]
        self.train_models["actor"] = self.actor_policy
        self.train_models["critic"] = self.critic_policy
    
    def initialize_model(self):
        return NotImplemented

    # return action, prob(action), value(action)
    def choose_action(self, state):
        return NotImplemented

    def value_state(self, state):
        return self.critic_policy(state)
    
    def _compute_GAE(self, rewards, dones, values, last_value):
        advs = []
        lastgaelam = 0
        next_value = last_value
        for r, done, v in zip(reversed(rewards), reversed(dones), reversed(values)):
            delta = r + self.gamma * next_value*(1 - done) - v
            adv = lastgaelam = delta + self.gamma * self.lam * (1 - done) * lastgaelam
            advs.insert(0, adv)
            next_value = v
        advs = torch.Tensor(advs).to(self.device) 
        targets = advs + values
        advs =(advs - advs.mean()) / (advs.std() + 1e-10)
        return targets, advs
    
    def _get_minibatchs(self, states, actions, advs, targets, values , old_logprobs):
        inds = np.arange(len(advs))
        mini_batchdatas = []
        np.random.shuffle(inds)
        mini_batch = (int)(len(advs) / self.NBATCH)
        nstates = []
        nactions = []
        nadvs = []
        ntargets = []
        nvalues = []
        nold_logprobs = []
        for start in range(0, len(advs), mini_batch):
            end = start + mini_batch
            mbinds = inds[start:end]
            batch_states = states[mbinds]
            batch_actions = torch.squeeze(actions)[mbinds]
            batch_advs = advs[mbinds]
            batch_targets = targets[mbinds]
            batch_values = values[mbinds]
            batch_logprobs = old_logprobs[mbinds]
            nstates.append(batch_states)
            nactions.append(batch_actions)
            nadvs.append(batch_advs)
            ntargets.append(batch_targets)
            nvalues.append(batch_values)
            nold_logprobs.append(batch_logprobs)
        return nstates, nactions, nadvs, ntargets, nvalues, nold_logprobs
    
    def process_n_trajectory(self, n_rollouts):
        traj_states = []
        traj_actions = []
        traj_advs = []
        traj_returns = []
        traj_values = []
        traj_old_logprobs = []
        traj_episodes = []
        # if memory
        for traj in n_rollouts:
            traj.to_tensor()
            traj_episodes.append(traj.episode)
            states = torch.stack(traj.states).detach().to(self.device)
            last_state = torch.FloatTensor(traj.last_state).detach().to(self.device)
            rewards  =  torch.FloatTensor(traj.rewards).detach().to(self.device)
            actions =  torch.squeeze(torch.stack(traj.actions)).detach().to(self.device)
            old_logprobs = torch.squeeze(torch.stack(traj.old_logprob)).detach().to(self.device)
            last_value = self.critic_policy(last_state)
            last_value = torch.squeeze(last_value) 
            values = self.critic_policy(states).detach()
            values = torch.squeeze(values)
            returns, advs  = self._compute_GAE(rewards, traj.dones, values, last_value)
            traj_states.append(states)
            traj_actions.append(actions)
            traj_advs.append(advs)
            traj_returns.append(returns)
            traj_values.append(values)
            traj_old_logprobs.append(old_logprobs)
        states = torch.cat(traj_states, dim = 0)
        actions = torch.cat(traj_actions, dim = 0)
        advs = torch.cat(traj_advs, dim = 0)
        returns = torch.cat(traj_returns, dim = 0)
        values = torch.cat(traj_values, dim = 0)
        old_logprobs = torch.cat(traj_old_logprobs, dim = 0)
        return states, actions, advs, returns, values, old_logprobs, traj_episodes

    def approximate_KL(self, mb_old_logprob, mb_new_logprob):
        with torch.no_grad():
            log_prob_ratio = mb_new_logprob - mb_old_logprob
            approximate_kl_div = torch.mean((torch.exp(log_prob_ratio) - 1) - log_prob_ratio).cpu().numpy()
            return approximate_kl_div

    def critic_loss(self, mb_obs, mb_returns, mb_old_values):
        mb_values = self.critic_policy(mb_obs)
        if len(mb_values.shape) == 2:
            mb_values = mb_values.squeeze(1)
        #loss1 = (mb_returns - mb_values).pow(2).mean()
        mb_targets = mb_old_values + torch.clamp(
                      mb_values - mb_old_values, -self.CLIP_EPSILON, self.CLIP_EPSILON
                    )
        #clipped critic loss
        clipped_critic_loss = F.mse_loss(mb_returns, mb_targets)
        # unclipped critic loss
        # critic_loss = F.mse_loss(mb_returns, mb_values)
        # critic_loss = 0.5*torch.mean(torch.max(critic_loss, clipped_critic_loss))
        return clipped_critic_loss
    
    def policy_loss(self, mb_old_logprobs, mb_obs, mb_actions, mb_advs):
        # to do calucate the policy clip here
        clip_range = self.CLIP_EPSILON
        try:
            mb_log_probs, entropy = self.actor_policy.module.evaluate(mb_obs, mb_actions)
        except:
            mb_log_probs, entropy = self.actor_policy.evaluate(mb_obs, mb_actions)
        # actor loss
        # imortant sampling
        prob_ratio = torch.exp(mb_log_probs - mb_old_logprobs)
        surr1 = prob_ratio * mb_advs
        surr2 = torch.clamp(prob_ratio, 1 - clip_range, 1 + clip_range) * mb_advs
        actor_loss = -torch.min(surr1, surr2).mean()
        # entropy loss
        entropy_loss = -torch.mean(entropy)
        return actor_loss, entropy_loss, mb_log_probs
    
    def train(self, memory):
        critic_loss_list = []
        policy_loss_list = []
        total_loss_list = []
        entropy_loss_list = []
        # get advantages, returns of the n trajectories
        states, actions, advs, returns, values, old_logprobs, traj_episode = self.process_n_trajectory(memory)
        self.traj_episode = max(traj_episode)
        # set up mini-batch
        nstates, nactions, nadvs, nreturns, nvalues, nold_logprobs = self._get_minibatchs(states,
                                                                                          actions, advs, returns, 
                                                                                          values, old_logprobs)
        # updated networks
        for i in range(self.TRAIN_EPOCH):
            for n in range(self.NBATCH):
                # sampling mini batch
                training = True
                mini_oldvalues = nvalues[n]
                mini_returns = nreturns[n].detach()
                mini_states = nstates[n].detach()
                mini_actions = nactions[n].detach()
                mini_advs = nadvs[n].detach()
                mini_old_logprobs = nold_logprobs[n].detach()
                self.critic_policy.train()
                self.actor_policy.train()
                # calculate policy loss and critic loss
                policy_loss, entropy_loss, mini_new_logprobs = self.policy_loss(mini_old_logprobs, mini_states, mini_actions, mini_advs)
                critic_loss = self.critic_loss(mini_states, mini_returns, mini_oldvalues)
                total_loss = policy_loss + self.critic_loss_coef * critic_loss + self.entropy_coef * entropy_loss
                policy_loss_list.append(policy_loss.item())
                critic_loss_list.append(critic_loss.item())
                total_loss_list.append(total_loss.item())
                entropy_loss_list.append(entropy_loss.item())
                # use approximated kl divergency to determine whether should keep training or not
                approx_kl_div = self.approximate_KL(mini_old_logprobs, mini_new_logprobs)
                # stop training due huge gap between 2 models
                if approx_kl_div > 1.5 * self.target_kl:
                    training = False
                    print("early stopping cuz reaching max kl div")
                    break
                self.critic_optimizer.zero_grad()
                self.actor_optimizer.zero_grad()
                total_loss.backward()
                # Clip grad norm
                torch.nn.utils.clip_grad_norm_(self.critic_policy.parameters(), self.max_grad_norm)
                torch.nn.utils.clip_grad_norm_(self.actor_policy.parameters(), self.max_grad_norm)
                self.critic_optimizer.step()
                self.actor_optimizer.step()
            if not training:
                break
        return np.mean(policy_loss_list), np.mean(critic_loss_list), np.mean(total_loss_list), np.mean(entropy_loss_list)
    
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

class Discrete_PPO_Agent(Base_PPO_Agent):
    def __init__(self, state_dim, action_dim, gamma = 0.99, lam = 0.95, critic_loss_coef = 0.5, TRAIN_EPOCH = 10, NBATCH = 1, CLIP_EPSILON = 0.2, entropy_coef = 0.00155, nopstepochs = 4, lr = 0.001, log_interval = 1, max_grad_norm = 0.75, target_kl = 0.75,use_critic_network = True, use_actor_network = True, device_id=0):
        super(Discrete_PPO_Agent, self).__init__(state_dim, action_dim, gamma=gamma, lam=lam, critic_loss_coef=critic_loss_coef, TRAIN_EPOCH = TRAIN_EPOCH, NBATCH = NBATCH, CLIP_EPSILON = CLIP_EPSILON, entropy_coef=entropy_coef, nopstepochs=nopstepochs, lr=lr, log_interval=log_interval, max_grad_norm=max_grad_norm, target_kl=target_kl, use_critic_network=use_critic_network, use_actor_network=use_actor_network, device_id=device_id)
        self.initialize_model()

    def initialize_model(self):
        if self.use_actor_network:
            self.actor_policy = Discrete_ActorModel(self.state_dim, self.action_dim)
            try:
                self.actor_policy.load_state_dict(torch.load(ACTOR_MODEL_PATH))
            except:
                path = "model"
                if not os.path.exists(path):
                    os.makedirs("model")
                print("cant find the model")
            self.actor_policy = self.actor_policy.to(self.device)
            self.actor_optimizer = self.optim(self.actor_policy.parameters(), lr=self.lr)
            self.train_models["actor"] = self.actor_policy

        if self.use_critic_network:
            self.critic_policy = Discrete_CriticModel(self.state_dim)
            try:
                self.critic_policy.load_state_dict(torch.load(CRITIC_MODEL_PATH))
            except:
                path = "model"
                if not os.path.exists(path):
                    os.makedirs("model")
                print("cant find the model")
            self.critic_policy = self.critic_policy.to(self.device)
            self.critic_optimizer = self.optim(self.critic_policy.parameters(), lr=self.lr*2)
            self.train_models["critic"] = self.critic_policy

    # override choose action
    def choose_action(self, state, eps = 0):
        # action, prob(action)
        state = state.to(self.device)
        probs = self.actor_policy.actor(state)
        m = Categorical(probs)
        action = m.sample()
        log_prob = m.log_prob(action)
        # value(action)
        #action_v = self.critic_policy(state) 
        return action, log_prob

class Continuous_PPO_Agent(Base_PPO_Agent):
    def __init__(self, state_dim, action_dim, max_action, gamma = 0.99, lam = 0.95, TRAIN_EPOCH = 10, NBATCH = 1, CLIP_EPSILON = 0.2, critic_loss_coef = 0.5, entropy_coef = 0.00155, nopstepochs = 4, lr = 0.001, log_interval = 1, max_grad_norm = 0.75, target_kl = 10,use_critic_network = True, use_actor_network = True, device_id=0):
        super(Continuous_PPO_Agent, self).__init__(state_dim, action_dim, gamma=gamma, lam=lam, TRAIN_EPOCH = TRAIN_EPOCH, NBATCH = NBATCH, CLIP_EPSILON = CLIP_EPSILON, critic_loss_coef=critic_loss_coef, entropy_coef=entropy_coef, nopstepochs=nopstepochs, lr=lr, log_interval=log_interval, max_grad_norm=max_grad_norm, target_kl=target_kl, use_critic_network=use_critic_network, use_actor_network=use_actor_network, device_id=device_id)
        self.max_action = max_action
        self.action_std = self.init_std = 0.8
        self.min_std = 0.001
        self.initialize_model()

    def initialize_model(self):
        if self.use_actor_network:
            self.actor_policy = Continous_ActorModel(self.state_dim, self.action_dim, self.max_action)
            try:
                self.actor_policy.load_state_dict(torch.load(ACTOR_MODEL_PATH))
            except EOFError as e:
                path = "model"
                if not os.path.exists(path):
                    os.makedirs("model")
                print("cant find the model")
            self.actor_policy = self.actor_policy.to(self.device)
            self.actor_optimizer = self.optim(self.actor_policy.parameters(), lr=self.lr)
            self.train_models["actor"] = self.actor_policy

        if self.use_critic_network:
            self.critic_policy = Continous_CriticModel(self.state_dim)
            try:
                self.critic_policy.load_state_dict(torch.load(CRITIC_MODEL_PATH))
            except EOFError as e:
                path = "model"
                if not os.path.exists(path):
                    os.makedirs("model")
                print("cant find the model")
            self.critic_policy = self.critic_policy.to(self.device)
            self.critic_optimizer = self.optim(self.critic_policy.parameters(), lr=self.lr*2)
            self.train_models["critic"] = self.critic_policy

    # override choose action
    def choose_action(self, state, eps):
       # action, prob(action)
        self.decay_action_std(eps)
        state = state.to(self.device)
        mu = self.actor_policy(state)
        std = self.action_std * torch.ones_like(mu)
        # policy distribution
        dist = Normal(mu, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).detach()
        return action.cpu(), log_prob.cpu()
    
    def decay_action_std(self, eps = 0):
        self.action_std = self.init_std - eps/80000.0
        if (self.action_std <= self.min_std):
            self.action_std = self.min_std
    
    def policy_loss(self, mb_old_logprobs, mb_obs, mb_actions, mb_advs):
        # notice: need to ensure the diff of action std between actor and learner is small enough, otherwise not converge
        self.decay_action_std(eps=self.traj_episode)
        # to do calucate the policy clip here
        clip_range = self.CLIP_EPSILON
        try:
            mb_log_probs, entropy = self.actor_policy.module.evaluate(mb_obs, mb_actions, self.action_std)
        except:
            mb_log_probs, entropy = self.actor_policy.evaluate(mb_obs, mb_actions, self.action_std)
        # actor loss
        # imortant sampling
        prob_ratio = torch.exp(mb_log_probs.sum(-1) - mb_old_logprobs.sum(-1))
        surr1 = prob_ratio * mb_advs
        surr2 = torch.clamp(prob_ratio, 1 - clip_range, 1 + clip_range) * mb_advs
        actor_loss = -torch.min(surr1, surr2).mean()
        # entropy loss
        entropy_loss = -torch.mean(entropy)
        return actor_loss, entropy_loss, mb_log_probs
