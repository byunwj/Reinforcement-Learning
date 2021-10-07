Reinforcement Learning
======================

This repository contains information (and code) for the two most popular approaches of reinforcement learning algorithms, Q-Learning and Policy Gradients. Note that DQN was implemented using Keras while DDPG and SAC were implemented at the Tensorflow level (i.e. TF v1). Updating the actor network based on the output of the critic network (to maximize the Q-value, which is the sum of future rewards) could have been implemented using Keras by using tf.GradientTape(), apply_gradients and such. However, the speed was noticeably slower compared to when implemented with TF v1.

# Q-Learning
## Deep Q-Networks (DQN)