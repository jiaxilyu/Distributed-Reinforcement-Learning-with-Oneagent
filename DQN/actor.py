from oneagent import Worker, service, logger, event_handler
import argparse
import torch
import gym
from policy.utils import PolicyFactory, get_env_kwargs
from oneagent import config

policy_name = config.policy
Batch = 3
class Actor(Worker):
    def __init__(self, **kwargs):
        super().__init__()
        self.policy = None
        self.eval_cnt = 1
        self.env_name = config.env_name
        self.policy_kwargs, self.ENV_A_SHAPE = self.get_policy_kwargs(kwargs)

    def get_policy_kwargs(self, kwargs):
        policy_kwargs, ENV_A_SHAPE = get_env_kwargs(self.env_name)
        policy_kwargs["eps_start"] = kwargs["EPS_START"]
        policy_kwargs["eps_min"] = kwargs["EPS_END"]
        policy_kwargs["eps_decay"] = kwargs["EPS_DECAY"]
        return policy_kwargs, ENV_A_SHAPE

    # batch up n times policy_eval queries from env
    @service(response=True, batch=Batch)
    def policy_eval(self, state):
        use_batch = True
        if type(state) != list:
                state = [state]
                use_batch = False
        if self.policy is not None:
            if type(state) != list:
                state = list(state)
            action = self.policy.choose_action(state, self.ENV_A_SHAPE)
            self.eval_cnt += 1
            if use_batch:
               return action.tolist()
            else:
               return action.tolist()[0]
        else:
            if use_batch:
                return [None for i in range(Batch)]
            else:
                return None

    @event_handler('policy_update')
    def handler(self, model):
        if self.policy is None:
           self.policy = PolicyFactory(kwargs=self.policy_kwargs, policy=policy_name)
        logger.info("successfully updated actor model")
        self.policy.model.load_state_dict(model)
        logger.info('recv msg, update policy')

