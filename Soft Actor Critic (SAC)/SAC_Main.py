# sean sungil kim
import numpy as np

import gym
import tensorflow as tf
from SAC_Networks import SAC
from SAC_Config import Config
from collections import deque
from matplotlib import pyplot as plt
import sys

# set the config class
config = Config()

if __name__ == "__main__":
    tf.compat.v1.reset_default_graph()
    tf.compat.v1.disable_eager_execution()
    env = gym.make('Pendulum-v0')
    deq = deque()

    # define the state and the action space size
    state_size = env.observation_space.shape[0]         #3
    action_size = env.action_space.shape[0]             #1

    # upper bound of the action space
    action_upper_bound = env.action_space.high[0]
    action_lower_bound = env.action_space.low[0]

    sess = tf.compat.v1.Session()
    alpha = 0.05
    sac = SAC(sess, config, state_size, action_size, action_lower_bound, action_upper_bound, alpha)

    step = 0
    return_lst = []
    training_started = None
    for episode in range(config.EPISODES):
        # for each episode, reset the environment
        state = env.reset()
        score = 0
        while True:
            if config.RENDER:
                env.render()

            # t
            # retrieve the action from the ddpg model
            action = sac.act(state)
            
            # input the action to the environment, and obtain the following
            next_state, reward, done, _ = env.step(action)
            
            # store it in the replay experience queue and go to the next state
            sac.remember(state, action, reward, next_state)

            # if there are enough instances in the replay experience queue, start the training
            if config.COUNTER > config.MEMORY_CAPACITY*1.1:
                if training_started == None:
                    training_started = episode
                sac.train()
                # actor_loss, critic_loss = ddpg.train()
                #print("Actor Loss: %.4f,\tCritic Loss: %.4f" % (actor_loss, critic_loss))
                #break
            # update the target_model every N steps
            #if step % config.TARGET_UPDATE_STEP == 0:
            #    ddpg.update_target_model()

            # t + 1
            # go to the next state
            state = next_state
            score += reward

            # if the episode is finished, go to the next episode
            if done:
                print("Episode: %i / %i,\tScore: %i,\tPointer: %i" % (episode, config.EPISODES, score, config.COUNTER))
                return_lst.append(score)
                break
        
        if np.mean(return_lst[-30:]) > -150:
            print("\n\t\t\t\t\t\tGoal Accomplished!\t\t\t\t\t\t\n")
    
    plt.plot(return_lst, linewidth=0.6, label='Episode Return')
    plt.axvline(training_started, linewidth=2, color="r", label='Training Phase Began')
    plt.legend()
    plt.title("Return over Episodes")
    plt.xlabel("Episodes")
    plt.ylabel("Return")
    plt.savefig("Return Graph (reward normalization).png")