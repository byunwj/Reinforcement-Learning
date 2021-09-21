# sean sungil kim
import numpy as np
from collections import deque
import sys
from matplotlib import pyplot as plt
import gym
import tensorflow as tf
from DDPG_Networks import DDPG
from DDPG_Config import Config


# set the config class
config = Config()

if __name__ == "__main__":
    tf.compat.v1.reset_default_graph()
    tf.compat.v1.disable_eager_execution()
    env = gym.make('Pendulum-v0')

    # define the state and the action space size
    state_size = env.observation_space.shape[0]         #3
    action_size = env.action_space.shape[0]             #1

    # upper bound of the action space
    action_upper_bound = env.action_space.high[0]       # 2
    action_lower_bound = env.action_space.low[0]        # -2

    # ddpg is specifically adapted for environments with continuous action spaces
    sess = tf.compat.v1.Session()
    ddpg = DDPG(sess, config, state_size, action_size, action_lower_bound, action_upper_bound)

    step = 0
    return_lst = []
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
                ddpg.train()
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
                print("Episode: %i / %i,\tScore: %i,\tPointer: %i,\tStandard Deviation: %.4f" % (episode, config.EPISODES, score, config.COUNTER, config.STAND_DEV))
                return_lst.append(score)
                break
        
        if np.mean(return_lst[-30:]) > -150:
            print("\n\t\t\t\t\t\tGoal Accomplished!\t\t\t\t\t\t\n")
            plt.plot(return_lst)
            plt.title("Return over Episodes")
            plt.xlabel("Episodes")
            plt.ylabel("Return")
            plt.savefig("Return Graph.png")
            sys.exit()
