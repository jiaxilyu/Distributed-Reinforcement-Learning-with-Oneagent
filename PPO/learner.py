from os import stat
from numpy import Inf
from oneagent import Worker, service, logger, EventDispatcher
from oneagent.utils import ObjectMemory
import argparse
import matplotlib.pyplot as plt
from oneagent.core.client import remote
from oneagent import remote, logger
from oneagent import config
from policy.utils import get_env_kwargs, PolicyFactory



class Learner(Worker):
    def __init__(self, **kwargs): # rank
        super().__init__()
        self.save_interval = kwargs["save_interval"]
        self.train_cnt = 0
        self.policy = None
        self.agent_type = config.agent_type
        self.env_name = config.env_name
        self.ed = EventDispatcher()
        self.policy_kwargs = self.get_policy_kwargs(kwargs)
        self.initialize_model(kwargs=self.policy_kwargs)
    
    # set up policy hyperparameter from yaml
    def get_policy_kwargs(self, kwargs):
        policy_kwargs = get_env_kwargs(self.env_name, self.agent_type)
        policy_kwargs["use_critic_network"] = True
        policy_kwargs["device_id"] = self.worker_id
        policy_kwargs["TRAIN_EPOCH"] = kwargs["TRAIN_EPOCH"]
        policy_kwargs["NBATCH"] = kwargs["NBATCH"]
        policy_kwargs["CLIP_EPSILON"] = kwargs["CLIP_EPSILON"]
        policy_kwargs["gamma"] = kwargs["gamma"]
        policy_kwargs["lam"] = kwargs["lam"]
        policy_kwargs["critic_loss_coef"] = kwargs["critic_loss_coef"]
        policy_kwargs["entropy_coef"] = kwargs["entropy_coef"]
        policy_kwargs["lr"] = kwargs["lr"]
        policy_kwargs["max_grad_norm"] = kwargs["max_grad_norm"]
        return policy_kwargs

    def initialize_model(self, kwargs):
        self.policy = PolicyFactory(typename=self.agent_type, kwargs=kwargs)
        network_weight = self.policy.actor_policy.state_dict()
        self.ed.fire("policy_update", payload= network_weight)
    
    @service(response=False)
    def train(self, memory):
        if type(memory) != list:
            memory = [memory]
        logger.info('---------------------------------train-----------------------------')
        policy_loss, critic_loss, total_loss, entropy_loss = self.policy.train(memory)
        network_weight = self.policy.actor_policy.state_dict()
        self.ed.fire("policy_update", payload= network_weight)
        logger.info("policy_loss: %s"%policy_loss)
        logger.info("critic_loss: %s"%critic_loss)
        logger.info("entory_loss: %s"%entropy_loss)
        logger.info("total_loss: %s"%total_loss)
        self.train_cnt += 1
        if self.train_cnt % self.save_interval == 0:
           self.policy.save_model()
