import os
import glob
import time
from datetime import datetime

import torch
import numpy as np

import gym

from PPO import PPO
#from minigrid.wrappers import FullyObsWrapper, RGBImgObsWrapper
from gymnasium.spaces.utils import flatten_space, flatten

import random
import pickle 
import gymnasium as gym

from automatica2025 import *
from pathlib import Path
from examples.mdp_pref.automatica2025.RL_gym_env import BeeRobotEnv
import matplotlib.pyplot as plt

#################################### Testing ###################################
def test():
    print("============================================================================================")

    ####### initialize environment hyperparameters ######
    with open(Path().absolute().parent /"automatica2025" /".tmp" / "model.pkl", "rb") as model_file:
        prod_game = pickle.load(model_file)

    with open(Path().absolute().parent /"automatica2025" / ".tmp" / "solutions.pkl", "rb") as model_file:
        solutions = pickle.load(model_file)
        solver = solutions[0]
    
    CONFIG = {
        "num_columns": 5,
        "num_rows": 4,
        "actions": ["N", "E", "S", "W", "Y", "T"],
        "bee_initial_loc": (1, 0),
        "bird_initial_loc": (3, 1),
        "battery_capacity": 12,
        "bird_bounds": {(2, 0), (2, 1), (3, 0), (3, 1), (4, 0), (4, 1)},
        "tulip_loc": (4, 3),
        "orchid_loc": (1, 1),
        "daisy_loc": (0, 2),
        "bee_dynamic_stochastic": False,
        "bee_dynamic_stochasticity_prob": 0.1,
        "spec_file_path": Path().parent.absolute() / "beerobot.prefltlf"
    }
    env = BeeRobotEnv(
        config=CONFIG,
        game=prod_game,
        solver=solver,
        render_mode="human",
    )

    has_continuous_action_space = False 
    max_ep_len = 15           # max timesteps in one episode
    action_std = 0.1            # set same std for action distribution which was used while saving

    render = False              # render environment on screen
    frame_delay = 0             # if required; add delay b/w frames

    total_test_episodes = 1000    # total num of testing episodes

    K_epochs = 80               # update policy for K epochs
    eps_clip = 0.2              # clip parameter for PPO
    gamma = 0.99                # discount factor

    lr_actor = 0.0003           # learning rate for actor
    lr_critic = 0.001           # learning rate for critic

    #####################################################

    # env = gym.make(env_name, max_steps = max_ep_len, render_mode="human")
    # # env = RGBImgObsWrapper(env)
    # env = FullyObsWrapper(env)
    # state space dimension
    env_name = "BeeRobotEnv"
    state_dim = env.observation_space.shape[0]

    # action space dimension
    if has_continuous_action_space:
        action_dim = env.action_space.shape[0]
    else:
        action_dim = env.action_space.n-1

    # initialize a PPO agent
    ppo_agent = PPO(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, has_continuous_action_space, action_std)

    # preTrained weights directory

    random_seed = 0             #### set this to load a particular checkpoint trained on random seed
    run_num_pretrained = 0      #### set this to load a particular checkpoint num

    directory = "PPO_preTrained" + '/' + env_name + '/'
    checkpoint_path = directory + "PPO_{}_{}_{}.pth".format(env_name, random_seed, run_num_pretrained)
    print("loading network from : " + checkpoint_path)

    ppo_agent.load(checkpoint_path)

    print("--------------------------------------------------------------------------------------------")

    test_running_reward = 0
    done_arr = []
    reward_arr = []

    for ep in range(1, total_test_episodes+1):
        ep_reward = 0
        state, info = env.reset()
        print("============================================================================================")
        print("ep {}: resetting env.".format(ep))
        print("initial state is: {}".format(state))
        print(info)
        for t in range(1, max_ep_len+1):
            # state['image'] = np.swapaxes(state['image'],0,2)
            # state['image'] = np.expand_dims(state['image'], axis=0)
            action = ppo_agent.select_action(state)    
            state, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            aut_state = None
            if "state" in info and info["state"] is not None:
                aut_state = info["state"].aut_state
            print(f"Action: {CONFIG['actions'][action]}, aut_state: {aut_state}, Reward: {reward}, Terminated: {terminated}")
            print(state)
            print(info)
            print("======================")
            # time.sleep(1)
            key=input("Press Enter to continue...")
            if terminated or truncated:
                break

        # clear buffer
        ppo_agent.buffer.clear()

        test_running_reward +=  ep_reward
        print('Episode: {} \t\t Reward: {}'.format(ep, round(ep_reward, 2)))
        ep_reward = 0

    env.close()

    print("============================================================================================")

    avg_test_reward = test_running_reward / total_test_episodes
    avg_test_reward = round(avg_test_reward, 2)
    print("average test reward : " + str(avg_test_reward))
    print("Success rate : ", np.mean(done_arr))

    print("============================================================================================")


if __name__ == '__main__':

    test()