import gym
from .PPO_Agent import *

# return kwargs of the RL policy
def get_env_kwargs(env_name, agent_type):
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    if agent_type == "Discrete":
       action_dim = env.action_space.n
       kwargs = {"state_dim":state_dim, "action_dim":action_dim}
       return kwargs
    else:
       action_dim = env.action_space.shape[0]
       max_action = float(env.action_space.high[0])
       kwargs = {"state_dim":state_dim, "action_dim":action_dim, "max_action":max_action}
       return kwargs

def PolicyFactory(kwargs, typename):
    if typename == "Discrete":
        return Discrete_PPO_Agent(**kwargs)
    else:
        return Continuous_PPO_Agent(**kwargs)
