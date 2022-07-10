from os import stat
from numpy import Inf
from oneagent import Worker, service, logger, EventDispatcher
import argparse
import matplotlib.pyplot as plt
from oneagent.core.client import remote
from oneagent import remote, logger
from oneagent import config
from policy.utils import get_env_kwargs, PolicyFactory
from collections import namedtuple
import os
from oneagent import config
import time

class Learner(Worker):
    def __init__(self, **kwargs): # rank
        super().__init__()
        self.save_interval = kwargs["save_interval"]
        self.train_cnt = 0
        self.policy = None
        self.policy_update_steps = 20
        self.ed = EventDispatcher()
        self.env_name = config.env_name
        self.policy_kwargs =self.get_policy_kwargs(kwargs)
        self.initialize_model(self.policy_kwargs)
        self.initime = self.endtime = time.time()
        # need to do, read from config
        self.train_interval = 10
        self.update_tar_interval = 100
    
    def get_policy_kwargs(self, kwargs):
        policy_kwargs = get_env_kwargs(self.env_name)
        policy_kwargs["gamma"] = kwargs["gamma"]
        policy_kwargs["lr"] = kwargs["lr"]
        policy_kwargs["buffer_size"] = kwargs["buffer_size"]
        policy_kwargs["batch_size"] = kwargs["batch_size"]
        policy_kwargs["tau"] = kwargs["tau"]
        return policy_kwargs

    def initialize_model(self, policy_kwargs):
         self.policy = PolicyFactory(kwargs=policy_kwargs)
         network_weight = self.policy.actor_policy.state_dict()
         self.ed.fire("policy_update", payload= network_weight)
         logger.info("initialize policy netork")
        
    @service(response=False, batch=5)
    def send_to_learner(self, experiences):
        result = []
        logger.info('---------------------------------train-----------------------------')
        Experience = namedtuple("Experience", ("state", "reward", "action", "next_state", "done"))
        for experience in experiences:
            experience = Experience(*experience)
            self.policy.store_experience(experience)
            result.append(None)
        #print(state, action, reward, next_state, done)
        logger.info('learn once')
        critic_loss, actor_loss = self.policy.train()
        logger.info("critic loss : %s"%critic_loss)
        logger.info("actor loss : %s"%actor_loss)
        self.train_cnt += 5
        if self.train_cnt % self.update_tar_interval == 0:
            network_weight = self.policy.actor_policy.state_dict()
            #self.policy.target_model.load_state_dict(policy)
            self.ed.fire("policy_update", payload=network_weight)
            logger.info('update policy model')
        if self.train_cnt % self.save_interval == 0:
            self.policy.save_model()
        return result