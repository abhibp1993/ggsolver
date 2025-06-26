import os
import glob
import time
from datetime import datetime

import torch
import numpy as np

from PPO import PPO
import gymnasium as gym
from gymnasium.spaces.utils import flatten_space, flatten
#from minigrid.wrappers import FullyObsWrapper, RGBImgObsWrapper

import random
import pickle 
import gymnasium as gym

from automatica2025 import *
from pathlib import Path
from examples.mdp_pref.automatica2025.RL_gym_env import BeeRobotEnv
import matplotlib.pyplot as plt

################################### Training ###################################
def train():
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

    has_continuous_action_space = False  # continuous action space; else discrete

    max_ep_len = 15                   # max timesteps in one episode
    max_training_timesteps = int(1e4)   # break training loop if timeteps > max_training_timesteps

    print_freq = max_ep_len * 10        # print avg reward in the interval (in num timesteps)
    log_freq = max_ep_len * 2           # log avg reward in the interval (in num timesteps)
    save_model_freq = int(1e4)          # save model frequency (in num timesteps)

    action_std = 0.6                    # starting std for action distribution (Multivariate Normal)
    action_std_decay_rate = 0.05        # linearly decay action_std (action_std = action_std - action_std_decay_rate)
    min_action_std = 0.1                # minimum action_std (stop decay after action_std <= min_action_std)
    action_std_decay_freq = int(2.5e5)  # action_std decay frequency (in num timesteps)
    #####################################################

    ## Note : print/log frequencies should be > than max_ep_len

    ################ PPO hyperparameters ################
    update_timestep = max_ep_len*6      # update policy every n timesteps
    K_epochs = 15               # update policy for K epochs in one PPO update

    eps_clip = 0.2          # clip parameter for PPO
    gamma = 0.99            # discount factor

    lr_actor = 0.0004       # learning rate for actor network
    lr_critic = 0.001       # learning rate for critic network

    random_seed = 0         # set random seed if required (0 = no random seed)
    #####################################################
    env_name = "BeeRobotEnv"
    print("training environment name : " + env_name)

    # env = gym.make(env_name, render_mode = "human")
    #env = gym.make(env_name, max_steps = max_ep_len)
    # env = RGBImgObsWrapper(env)
    #env = FullyObsWrapper(env)


    # state space dimension
    state_dim = env.observation_space.shape[0]
    
    # action space dimension
    if has_continuous_action_space:
        action_dim = env.action_space.shape[0]
    else:
        action_dim = env.action_space.n-1
    
    print(f"State_dim: {state_dim} Action_dim: {action_dim}")

    ###################### logging ######################

    #### log files for multiple runs are NOT overwritten
    log_dir = "PPO_logs"
    if not os.path.exists(log_dir):
          os.makedirs(log_dir)

    log_dir = log_dir + '/' + env_name + '/'
    if not os.path.exists(log_dir):
          os.makedirs(log_dir)

    #### get number of log files in log directory
    run_num = 0
    current_num_files = next(os.walk(log_dir))[2]
    run_num = len(current_num_files)

    #### create new log file for each run
    log_f_name = log_dir + '/PPO_' + env_name + "_log_" + str(run_num) + ".csv"

    print("current logging run number for " + env_name + " : ", run_num)
    print("logging at : " + log_f_name)
    #####################################################

    ################### checkpointing ###################
    run_num_pretrained = 0      #### change this to prevent overwriting weights in same env_name folder

    directory = "PPO_preTrained"
    if not os.path.exists(directory):
          os.makedirs(directory)

    directory = directory + '/' + env_name + '/'
    if not os.path.exists(directory):
          os.makedirs(directory)


    checkpoint_path = directory + "PPO_{}_{}_{}.pth".format(env_name, random_seed, run_num_pretrained)
    print("save checkpoint path : " + checkpoint_path)
    #####################################################


    ############# print all hyperparameters #############
    print("--------------------------------------------------------------------------------------------")
    print("max training timesteps : ", max_training_timesteps)
    print("max timesteps per episode : ", max_ep_len)
    print("model saving frequency : " + str(save_model_freq) + " timesteps")
    print("log frequency : " + str(log_freq) + " timesteps")
    print("printing average reward over episodes in last : " + str(print_freq) + " timesteps")
    print("--------------------------------------------------------------------------------------------")
    print("state space dimension : ", state_dim)
    print("action space dimension : ", action_dim)
    print("--------------------------------------------------------------------------------------------")
    if has_continuous_action_space:
        print("Initializing a continuous action space policy")
        print("--------------------------------------------------------------------------------------------")
        print("starting std of action distribution : ", action_std)
        print("decay rate of std of action distribution : ", action_std_decay_rate)
        print("minimum std of action distribution : ", min_action_std)
        print("decay frequency of std of action distribution : " + str(action_std_decay_freq) + " timesteps")
    else:
        print("Initializing a discrete action space policy")
    print("--------------------------------------------------------------------------------------------")
    print("PPO update frequency : " + str(update_timestep) + " timesteps")
    print("PPO K epochs : ", K_epochs)
    print("PPO epsilon clip : ", eps_clip)
    print("discount factor (gamma) : ", gamma)
    print("--------------------------------------------------------------------------------------------")
    print("optimizer learning rate actor : ", lr_actor)
    print("optimizer learning rate critic : ", lr_critic)
    if random_seed:
        print("--------------------------------------------------------------------------------------------")
        print("setting random seed to ", random_seed)
        torch.manual_seed(random_seed)
        env.seed(random_seed)
        np.random.seed(random_seed)
    #####################################################

    print("============================================================================================")

    ################# training procedure ################

    # initialize a PPO agent
    ppo_agent = PPO(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, has_continuous_action_space, action_std)

    # track total training time
    start_time = datetime.now().replace(microsecond=0)
    print("Started training at (GMT) : ", start_time)

    print("============================================================================================")

    # logging file
    log_f = open(log_f_name,"w+")
    log_f.write('episode,timestep,reward\n')

    # printing and logging variables
    print_running_reward = 0
    print_running_episodes = 0

    log_running_reward = 0
    log_running_episodes = 0

    time_step = 0
    i_episode = 0

    done_arr = []
    reward_arr = []
    # training loop
    while time_step <= max_training_timesteps:

        state, info = env.reset()
        #print(f"Passed in state {state}")
        current_ep_reward = 0
        print_avg_reward = 0

        for t in range(1, max_ep_len+1):

            # select action with policy
            # state['image'] = np.swapaxes(state['image'],0,2)
            # state['image'] = np.expand_dims(state['image'], axis=0)
            action = ppo_agent.select_action(state)    
            state, reward, terminated, truncated, info = env.step(action)

            # saving reward and is_terminals
            ppo_agent.buffer.rewards.append(reward)
            if terminated or truncated:
                ppo_agent.buffer.is_terminals.append(True)
            else:
                ppo_agent.buffer.is_terminals.append(False)
            # if truncated:
            #     ppo_agent.buffer.is_terminals.append(0)

            time_step +=1
            current_ep_reward += reward

            # update PPO agent
            if time_step % update_timestep == 0:
                ppo_agent.update()

            # if continuous action space; then decay action std of ouput action distribution
            if has_continuous_action_space and time_step % action_std_decay_freq == 0:
                ppo_agent.decay_action_std(action_std_decay_rate, min_action_std)

            # log in logging file
            if time_step % log_freq == 0:

                # log average reward till last episode
                log_avg_reward = log_running_reward / log_running_episodes
                log_avg_reward = round(log_avg_reward, 4)

                log_f.write('{},{},{}\n'.format(i_episode, time_step, log_avg_reward))
                log_f.flush()

                log_running_reward = 0
                log_running_episodes = 0

            # printing average reward
            if time_step % print_freq == 0:

                # print average reward till last episode
                print_avg_reward = print_running_reward / print_running_episodes
                print_avg_reward = round(print_avg_reward, 2)

                print("Episode : {} \t\t Timestep : {} \t\t Average Reward : {}".format(i_episode, time_step, print_avg_reward))

                print_running_reward = 0
                print_running_episodes = 0

                # removed 6/5
                # if print_avg_reward > 0.92:
                #     break

            # save model weights
            if time_step % save_model_freq == 0:
                print("--------------------------------------------------------------------------------------------")
                print("saving model at : " + checkpoint_path)
                ppo_agent.save(checkpoint_path)
                print("model saved")
                print("Elapsed Time  : ", datetime.now().replace(microsecond=0) - start_time)
                print("--------------------------------------------------------------------------------------------")

            # break; if the episode is over
            if terminated:
                done_arr.append(1)
                reward_arr.append(current_ep_reward)
                break
            if truncated:
                done_arr.append(0)
                reward_arr.append(current_ep_reward)
        #removed 6/5
        # if print_avg_reward > 0.92:
        #     print(print_avg_reward)
        #     print("saving converged model at : " + checkpoint_path)
        #     ppo_agent.save(checkpoint_path)
        #     print("model converged")            
        #     break

        print_running_reward += current_ep_reward
        print_running_episodes += 1

        log_running_reward += current_ep_reward
        log_running_episodes += 1

        i_episode += 1
        # if len(done_arr) > 50 and np.mean(done_arr[-10:])>0.9:
        #     print("saving converged model at : " + checkpoint_path)
        #     ppo_agent.save(checkpoint_path)
        #     print("model converged")            
        #     break
    log_f.close()
    env.close()

    # Plot rewards per episode
    plt.figure(figsize=(10, 6))
    plt.plot(reward_arr, label="Reward per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Reward per Episode During Training")
    plt.legend()
    plt.grid()
    plt.savefig("PPO_logs/BeeRobotEnv/reward_per_episode.png")  # Save the plot
    plt.show()
    print("Reward plot saved to PPO_logs/BeeRobotEnv/reward_per_episode.png")

    # print total training time
    print("============================================================================================")
    end_time = datetime.now().replace(microsecond=0)
    print("Started training at (GMT) : ", start_time)
    print("Finished training at (GMT) : ", end_time)
    print("Total training time  : ", end_time - start_time)
    print("============================================================================================")


if __name__ == '__main__':

    train()