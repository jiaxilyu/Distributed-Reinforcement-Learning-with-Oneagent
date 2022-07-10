from oneagent import Worker, service, logger, event_handler
from oneagent import config
import argparse
from policy.utils import PolicyFactory, get_env_kwargs
import torch

class Actor(Worker):
    def __init__(self, **kwargs):
        super().__init__()
        self.policy = None
        self.env_name = config.env_name
        self.policy_kwargs =self.get_policy_kwargs(self.env_name)
        self.eval_cnt = 1
    
    def get_policy_kwargs(self, env_name):
        policy_kwargs = get_env_kwargs(env_name)
        policy_kwargs["use_critic_network"] = False
        policy_kwargs["use_target"] = False
        return policy_kwargs

    @service()
    def policy_eval(self, state):
        if self.policy is not None:
            #epsilon = epsilon_by_frame(self.eval_cnt)
            state = torch.FloatTensor(state)
            action = self.policy.choose_action(state)
            self.eval_cnt += 1
            return action

    @event_handler('policy_update')
    def handler(self, model):
        if self.policy is None:
           self.policy = PolicyFactory(kwargs=self.policy_kwargs)
        logger.info("successfully updated actor model")
        self.policy.actor_policy.load_state_dict(model)
        logger.info('recv msg, update policy')