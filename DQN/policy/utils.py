from .DDQN_Agent import DDQN_Agent
from .DQN_Agent import DQN_Agent
from .Dueling_DQN_Agent import Dueling_DQN_Agent
from oneagent import config
from .model import *
import gym

CHECKPOINT = config.CHECKPOINT

def get_env_kwargs(env_name):
    env = gym.make(env_name)
    action_dim = env.action_space.n
    state_dim = env.observation_space.shape[0]
    kwargs = {"state_dim":state_dim, "action_dim":action_dim}
    ENV_A_SHAPE = 0 if isinstance(env.action_space.sample(), int) else env.action_space.sample().shape
    return kwargs, ENV_A_SHAPE

def PolicyFactory(kwargs, policy="DQN"):
    kwargs["check_point_path"] = CHECKPOINT
    kwargs["policy_name"] = policy
    if policy == "DQN":
        kwargs["model_class"] = DQN_MODEL
        return DQN_Agent(**kwargs)
    elif policy == "DDQN":
        kwargs["model_class"] = DQN_MODEL
        return DDQN_Agent(**kwargs)
    elif policy == "DuelingDQN":
        kwargs["model_class"] = DuelingDQNModel
        return Dueling_DQN_Agent(**kwargs)
    else:
        raise Exception("policy not found")
