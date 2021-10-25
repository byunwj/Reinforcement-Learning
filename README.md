Reinforcement Learning
======================

This repository contains information (and code) for the two most popular approaches of reinforcement learning algorithms, Q-Learning and Policy Gradients (or rather Q-Learning + Policy Gradients). In **very** Simple words, Q-learning aims to learn the reward of a certain action (single deterministic) in a certain state and take a greedy action accordingly while policy gradient method aims to directly learn the best action (stochastic) in a certain state.

Note that DQN was implemented using Keras while DDPG and SAC were implemented at the Tensorflow level (i.e. TF v1). Updating the actor network based on the output of the critic network (to maximize the Q-value, which is the sum of future rewards) could have been implemented using Keras by using tf.GradientTape(), apply_gradients and such. However, the speed was noticeably slower compared to when implemented with TF v1.

# Policy Gradient + Q-Learning
## Deep Deterministic Policy Gradient (DDPG)

## Soft Actor Critic (SAC)



# Q-Learning
## Deep Q-Networks (DQN)

