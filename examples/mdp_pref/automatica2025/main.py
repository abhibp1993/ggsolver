import gym
from PG import SimplePG
import matplotlib.pyplot as plt
import pickle 
import numpy as np
import time
from PIL import Image
import pandas as pd
import seaborn as sns
import os 
import random

import glob
from datetime import datetime

import torch

from PPO import PPO
import gymnasium as gym
from gymnasium.spaces.utils import flatten_space, flatten
#from minigrid.wrappers import FullyObsWrapper, RGBImgObsWrapper

import pickle 
import gymnasium as gym

from automatica2025 import *
from pathlib import Path
from examples.mdp_pref.automatica2025.RL_gym_env import BeeRobotEnv

def plot_rewards(rewards_history, filename="rewards_plot.png"):
    save_dir = "PPO_logs/BeeRobotEnv"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, filename)

    plt.figure(figsize=(10, 6))
    plt.plot(rewards_history, label="Running Reward")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Reward per Episode")
    plt.legend()
    plt.grid()
    plt.savefig(save_path)  # Save the plot to the specified directory
    plt.show()
    print(f"Plot saved to {save_path}")

def main_dqn(episodes=10000, timesteps=50, dirname="results", random_seed=42):
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
    actionCnt = env.action_space.n
    D = env.observation_space.shape[0] # how many input neurons

    print("ACTION count IS: " + str(actionCnt))
    print("OBS Size IS: " + str(D))

    NUM_HIDDEN = 10
    GAMMA = 0.95
    #LEARNING_RATE = 1e-3
    LEARNING_RATE = 1e-2
    DECAY_RATE = 0.99
    MAX_EPSILON = 1.0
    MIN_EPSILON = 0.05
    EPSILON_DECAY = 0.995
    
    agent = SimplePG(actionCnt,D,NUM_HIDDEN,LEARNING_RATE,GAMMA,DECAY_RATE,MAX_EPSILON,random_seed)
    agent.set_explore_epsilon(MAX_EPSILON)
    reward_sum = 0
    running_reward = None
    rewards_history = []
	# keeps track of reward during evaluation
    total_eval_rewards = []
    aut_states = []
    for e in range(episodes):
        state,info = env.reset()
        print("============================================================================================")
        print("ep %f: resetting env." % e)
        print("initial state is: %s" % str(state))
        reward_sum = 0
        # print('ep %f: resetting env.' % e)
        # print('state is: %s' % str(state))
        for t in range(timesteps):
            action = agent.process_step(state,True)
			
            state, reward, terminated, truncated, info = env.step(action)
            print('ep %f: step %f, action %d, reward %f, new state %s' % (e, t, action, reward, str(state)))
            print('terminated is %s, info is %s' % (terminated, info))
            aut_state = None
            if "state" in info and info["state"] is not None:
                aut_state = info["state"].aut_state
                aut_states.append(aut_state)
            agent.give_reward(reward)
            reward_sum += reward
            if terminated is True:
                running_reward = reward_sum if running_reward is None else running_reward * 0.95 + reward_sum * 0.05
                rewards_history.append(running_reward)
                #print('ep %f: resetting env. episode reward total was %f. episode time steps was %f. running mean: %f' % (e, reward_sum, t, running_reward))
                agent.finish_episode()
				# update after every k episodes
                if e % 5 == 0:
                    agent.update_parameters()
                break
        if (e + 1) % 100 == 0:
            print(f"Episode {e + 1}: Running reward = {running_reward}")
            # Save the model every 100 episodes
            # model_save_path = f"{dirname}/model_episode_{e + 1}.pt"
            # torch.save(agent.model.state_dict(), model_save_path)
            # print(f"Model saved to {model_save_path}")
    print("============================================================================================")
    print("Training finished.")
    print(set(aut_states))
    print("Ranks visited:")
    print(set(env.ranks_visited))
    print("============================================================================================")
    #print(rewards_history)
    plot_rewards(rewards_history)


if __name__ == "__main__":
    main_dqn()