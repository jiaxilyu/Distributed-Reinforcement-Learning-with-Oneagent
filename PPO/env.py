from oneagent import remote, logger
import numpy as np
import torch
#vfrom oneagent.utils.atari_wrappers import make_atari, wrap_Framestack
from oneagent import config
import gym
import os
import time
from os.path import join as joindir
from os import makedirs as mkdir
import pandas as pd
from policy.memory import Memory


logger.set_level("info")

RESULT_DIR = joindir("../result", ".".join(__file__.split(".")[:-1]))
mkdir(RESULT_DIR, exist_ok=True)


@remote
def policy_eval():
    pass

@remote
def train():
    pass


def ensured_policy_eval(state, eps):
    while True:
        data = policy_eval([state, eps])
        if data is not None:
            return data
        else:
            time.sleep(1)


#env = make_atari(args.env_id) if args.use_wrapper else gym.make(args.env_id)
#env = wrap_Framestack(
    #env, scale=False, frame_stack=4) if args.use_wrapper else env
env = gym.make(config.env_name) 
env.seed(543)
frame = env.reset()

"""
(
    frame._force().transpose(2, 0, 1).shape
    if not args.mlp
    else env.observation_space.shape
)
"""

use_buffer_service = True

print_interval = 1000
episode_num = 0
episode_reward = 0
all_rewards = []
reward_record = []
capacity = []
losses = []


frames = 20000000
# avg_reward = 0
start = g_start = time.time()
memory = Memory()
update_step = 2048
step = 0
running_reward = []

start = time.time()
for i_episode in range(1, 1000000):
    state = env.reset()
    state = torch.FloatTensor(state)
    ep_reward = 0
    for t in range(1, 1000):  
        action, log_prob = ensured_policy_eval(state, i_episode)
        try:
            next_frame, reward, done, info = env.step(action.item())
        except:
            next_frame, reward, done, info = env.step(action.numpy())
        ep_reward += reward
        next_state = torch.FloatTensor(next_frame)
        memory.add(reward=reward, state=state, done=done, action=action, old_logprob=log_prob)
        state = next_state
        step += 1
        if step % update_step == 0:
            memory.last_state = next_state
            memory.episode = i_episode
            train(memory)
            memory.clear()
            #step = 0
        if step % 1000 == 0:
            end = time.time()
            time_consumption = end - start
            print("The time consumption for 10000 frames : %s"%time_consumption)
            start = time.time()
        if done:
            break
    running_reward.append(ep_reward)
    if i_episode % 100 == 0:
            print('Episode {}\tLast reward: {:.2f}\tAverage reward: {:.2f}'.format(
                  i_episode, ep_reward, np.mean(running_reward)))
            running_reward = []