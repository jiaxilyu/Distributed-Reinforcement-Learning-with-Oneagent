from oneagent import remote, logger
import numpy as np
import torch
import gym
import os
import time
from os.path import join as joindir
from os import makedirs as mkdir
import pandas as pd
import numpy as np
from oneagent import config


logger.set_level("info")

RESULT_DIR = joindir("../result", ".".join(__file__.split(".")[:-1]))
mkdir(RESULT_DIR, exist_ok=True)





@remote
def policy_eval():
    pass

@remote
def send_to_learner():
    pass


def ensured_policy_eval(state):
    while True:
        data = policy_eval(state)
        if data is not None:
            return data
        else:
            time.sleep(1)


DIR = os.path.abspath(os.path.dirname(__file__) +
                      os.path.sep + "dqn_model.pkl")

#env = make_atari(args.env_id) if args.use_wrapper else gym.make(args.env_id)
#env = wrap_Framestack(
    #env, scale=False, frame_stack=4) if args.use_wrapper else env
env = gym.make(config.env_name) 
env.seed(543)
frame = env.reset()

frames = 20000000
# avg_reward = 0
start = g_start = time.time()
step = 0
running_reward = []
max_episode = 10000
start = time.time()
for i_episode in range(1, max_episode):
    state = env.reset()
    #state = torch.FloatTensor(state)
    ep_reward = 0
    for t in range(1, 1000):
        # actor choose action 
        action = ensured_policy_eval(state)
        # step action on env
        next_state, reward, done, info = env.step(action)
        # add reward to sum reward
        ep_reward += reward
        # if we are training the agent
        if not config.eval_model:
            # experience = Experience(*[state, reward, action, next_state, 0 if done else 1])
            # send this experience to learner to learn
            send_to_learner([state, reward, action, next_state, 0 if done else 1])
        state = next_state
        step += 1
        if step % 1000 == 0:
            end = time.time()
            time_consumption = end - start
            print("The time consumption for 1000 frames : %s, speed : %s"%(time_consumption, 1000/time_consumption))
            start = time.time()
        if done:
            break
    running_reward.append(ep_reward)    
    # check reward progress
    if i_episode % 100 == 0:
            print('Episode {}\tLast reward: {:.2f}\tAverage reward: {:.2f}'.format(
                  i_episode, ep_reward, np.mean(running_reward)))