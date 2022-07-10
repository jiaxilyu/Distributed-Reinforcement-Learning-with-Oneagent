from os import stat
from numpy import Inf
from oneagent import Worker, service, logger, EventDispatcher
from oneagent.utils import ObjectMemory
import argparse
import matplotlib.pyplot as plt
from oneagent.core.client import remote
import torch
from policy.utils import PolicyFactory, get_env_kwargs
import time
import sys
import traceback
from oneagent import remote, logger
from oneagent import config
import gym
from collections import namedtuple

policy_name = config.policy

class Learner(Worker):
    def __init__(self, **kwargs): # rank
        super().__init__()
        self.train_cnt = 0
        self.policy = None
        self.policy_update_steps = 20
        self.save_interval = config.save_interval
        self.env_name = config.env_name
        self.kwargs = kwargs
        self.ed = EventDispatcher()
        self.policy_kwargs, self.ENV_A_SHAPE = self.get_policy_kwargs(kwargs)
        self.initialize_model(kwargs=self.policy_kwargs)
        self.initime = self.endtime = time.time()
        # need to do, read from config
        self.train_interval = 10
        self.update_tar_interval = 100
    
    def get_policy_kwargs(self, kwargs):
        policy_kwargs, ENV_A_SHAPE = get_env_kwargs(self.env_name)
        policy_kwargs["batch_size"] = self.kwargs["BATCH_SIZE"]
        policy_kwargs["lr"] = self.kwargs["LR"]
        policy_kwargs["gamma"] = self.kwargs["GAMMA"]
        policy_kwargs["target_policy_updated_step"] = self.kwargs["TARGET_REPLACE_ITER"]
        policy_kwargs["memory_buffer_size"] = self.kwargs["MEMORY_CAPACITY"]
        policy_kwargs["eps_start"] = self.kwargs["EPS_START"]
        policy_kwargs["eps_min"] = self.kwargs["EPS_END"]
        policy_kwargs["eps_decay"] = self.kwargs["EPS_DECAY"]
        return policy_kwargs, ENV_A_SHAPE

    def initialize_model(self, kwargs):
        # initialize policy
        self.policy = PolicyFactory(kwargs, policy=policy_name)
        network_weight = self.policy.model.state_dict()
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
        loss = self.policy.train()
        logger.info(loss)
        self.train_cnt += 5
        if self.train_cnt % self.save_interval == 0:
            self.policy.save_model()
        if self.train_cnt % self.update_tar_interval == 0:
            network_weight = self.policy.model.state_dict()
            self.policy.soft_updated()
            self.ed.fire("policy_update", payload=network_weight)
            logger.info('update policy model')
        return result
