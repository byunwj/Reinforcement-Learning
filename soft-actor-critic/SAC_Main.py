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
    for episode in range(config.EPISODES):
        # for each episode, reset the environment
        state = env.reset()
        score = 0
        while True:
            if config.RENDER:
                env.render()
            
            # retrieve the action from the sac model
            action = sac.act(state)

            # input the action to the environment, and obtain the following
            next_state, reward, done, _ = env.step(action)
            
            # store it in the replay experience queue and go to the next state
            sac.remember(state, action, reward, next_state)

            # if there are enough instances in the replay experience queue, start the training
            if config.COUNTER > config.MEMORY_CAPACITY:
                mu, std = sac.train()
                #print("mu: %.4f,\t stdev: %.4f" % (mu, std))

            # go to the next state
            state = next_state
            score += reward

            # if the episode is finished, go to the next episode
            if done:
                print("Episode: %d / %d,\tScore: %d,\tPointer: %d" % (episode, config.EPISODES, score, config.COUNTER))
                #if config.COUNTER > config.MEMORY_CAPACITY:
                    #print("mu: %f, \tstdev: %f, \talpha: %f" % (mu, std, alpha))
                deq.append(score)
                break
    
    
    plt.plot(deq)
    plt.title("SAC")
    plt.xlabel("Episodes")
    plt.ylabel("Score")
    plt.show()


    last_hundred = np.array([deq.pop() for i in range(100)])
    print("Average Score in the last 100 Episodes:", np.mean(last_hundred))
    
    