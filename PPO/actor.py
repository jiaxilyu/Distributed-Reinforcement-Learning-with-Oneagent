from oneagent import Worker, service, logger, event_handler
import torch
import gym
from oneagent import config
from policy.utils import get_env_kwargs, PolicyFactory

class Actor(Worker):
    def __init__(self, **kwargs):
        super().__init__()
        self.policy = None
        self.eval_cnt = 1
        self.env_name = config.env_name
        self.agent_type = config.agent_type
        self.policy_kwargs = self.get_policy_kwargs(kwargs)
    
    # set up policy hyperparameter from yaml
    def get_policy_kwargs(self, kwargs):
        policy_kwargs = get_env_kwargs(self.env_name, self.agent_type)
        policy_kwargs["use_critic_network"] = False
        policy_kwargs["device_id"] = self.worker_id
        return policy_kwargs

    @service(response=True)
    def policy_eval(self, msg):
        if self.policy is not None:
            #epsilon = epsilon_by_frame(self.eval_cnt)
            state, eps = msg
            action, logproba = self.policy.choose_action(state, eps)
            self.eval_cnt += 1
            logger.info("take action %s"%action)
            return action, logproba

    @event_handler('policy_update')
    def handler(self, model):
        if self.policy is None:
           self.policy = PolicyFactory(typename=self.agent_type, kwargs=self.policy_kwargs)
        logger.info("successfully updated actor model")
        self.policy.actor_policy.load_state_dict(model)
        logger.info('recv msg, update policy')
