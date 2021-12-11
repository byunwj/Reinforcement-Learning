import numpy as np
from collections import deque
import sys
from matplotlib import pyplot as plt
from tensorflow.python import training
import gym
import tensorflow as tf
from DDPG_Networks import DDPG
from DDPG_Config import Config


# set the config class
def run_main():
    config = Config()
    tf.compat.v1.reset_default_graph()
    tf.compat.v1.disable_eager_execution()
    env = gym.make('Pendulum-v0')

    # define the state and the action space size
    state_size = env.observation_space.shape[0]         #3
    action_size = env.action_space.shape[0]             #1

    # upper bound of the action space
    action_upper_bound = env.action_space.high[0]
    action_lower_bound = env.action_space.low[0]

    sess = tf.compat.v1.Session()
    ddpg = DDPG(sess, config, state_size, action_size, action_lower_bound, action_upper_bound)

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
            action = ddpg.act(state)
            
            # input the action to the environment, and obtain the following
            next_state, reward, done, _ = env.step(action)
            
            # store it in the replay experience queue and go to the next state
            ddpg.remember(state, action, reward, next_state)

            # if there are enough instances in the replay experience queue, start the training
            if config.COUNTER > config.MEMORY_CAPACITY:
                if training_started == None:
                    training_started = episode
                ddpg.train()

            # go to the next state
            state = next_state
            score += reward

            # if the episode is finished, go to the next episode
            if done:
                if len(return_lst) > 30:
                    print("Episode: %i / %i,\tScore: %i,\tPointer: %i, \tMean Return for last 30: %i" % (episode, config.EPISODES, score, config.COUNTER, round(np.mean(return_lst[-30:])) ))
                else:
                    print("Episode: %i / %i,\tScore: %i,\tPointer: %i" % (episode, config.EPISODES, score, config.COUNTER))
                return_lst.append(score)
                break
        
        if np.mean(return_lst[-30:]) > -150:
            print("\n\n\t\t\t\tGoal Accomplished!\n\n")

    plotting(return_lst, training_started)
    

def plotting(return_lst, training_started):
    plt.plot(return_lst, color = 'blue', linewidth=0.6, label='Episode Return')

    #plt.fill_between( [i for i in range(len(return_lst))],   
    #                  return_lst + np.std(return_lst)/2, 
    #                  return_lst - np.std(return_lst)/2,
    #                  color = 'blue', alpha = 0.3 )
  
    plt.axvline(training_started, linewidth=2, color="r", label='Training Phase Began')
    plt.axhline(np.array(return_lst[-200:]).mean(), color = "orange", label = 'Mean Return of Last 200 Episodes: {}'.format(round(np.array(return_lst[-200:]).mean())))
    plt.legend(loc = 'lower right')
    plt.title("Return over Episodes")
    plt.xlabel("Episodes")
    plt.ylabel("Return")
    title = 'Return Graph DDPG'
    plt.savefig(title)

if __name__ == "__main__":
    run_main()
