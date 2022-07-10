import gym
from .DDPG_Agent import *

# return kwargs of the RL policy
def get_env_kwargs(env_name):
    env = gym.make(env_name) 
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    a_high_bound = env.action_space.high
    a_low_bound = env.action_space.low
    kwargs = {"state_dim":state_dim, "action_dim":action_dim, "action_low_bound":a_low_bound, "action_high_bound":a_high_bound}
    return kwargs

def PolicyFactory(kwargs):
    return DDPG_Agent(**kwargs)
