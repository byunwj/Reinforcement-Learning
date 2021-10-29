Reinforcement Learning
======================

This repository contains information (and code) for the two most popular approaches of reinforcement learning algorithms, Deep Q-Learning and Policy Gradients (or rather Q-Learning + Policy Gradients). In **very** Simple words, Q-learning aims to learn the reward of a certain action (single deterministic) in a certain state and take a greedy action accordingly while policy gradient method aims to directly learn the best action (stochastic) in a certain state.

Note that DQN was implemented using Keras while DDPG and SAC were implemented at the Tensorflow level (i.e. TF v1). Updating the actor network based on the output of the critic network (to maximize the Q-value, which is the sum of future rewards) could have been implemented using Keras by using tf.GradientTape(), apply_gradients and such. However, the speed was noticeably slower compared to when implemented with TF v1.

# Q-Learning
 While in Markov Process (MP), the transition probability dictates the next state from one state, in MDP, action dictates the next state from one state. And there ares consequences of taking a certain action in every state, which we call "rewards". The idea of Q-learning is to learn the optimal policy in a Markov Decision Process (MDP). 

# Deep Q-Learning
## Deep Q-Networks (DQN)
DQN is a reinforcement learning algorithm that combines Q-learning and deep neural networks. DQN approximates a state-value function in the Q-learning framework with a neural network. It utilizes off-policy learning by using Experience Replay from which random samples (could be prioritized) are drawn for training. Experience Replay usually contains a specified number of recent (state, action, next state, reward). 

# Policy Gradients + Q-Learning
## Deep Deterministic Policy Gradient (DDPG)

## Soft Actor Critic (SAC)





