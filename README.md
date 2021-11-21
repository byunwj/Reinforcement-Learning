Reinforcement Learning
======================

This repository contains a brief explanation (and code) of the two most well-known approaches of model-free reinforcement learning algorithms, Deep Q-Learning and Policy Gradients (or rather Q-Learning + Policy Gradients). In **very** Simple words, Q-learning aims to learn the reward of a certain action (single deterministic) in a certain state and take a greedy action accordingly while policy gradient method aims to directly learn the best action (stochastic) in a certain state.

Note that DQN was implemented using Keras while DDPG and SAC were implemented at the Tensorflow level (i.e. TF v1). Updating the actor network based on the output of the critic network (to maximize the Q-value, which is the sum of future rewards) could have been implemented using Keras by using tf.GradientTape(), apply_gradients and such. However, the speed was noticeably slower compared to when implemented with TF v1.


# Q-Learning
 While in Markov Process (MP), the transition probability dictates the next state from one state, in MDP, the action dictates the next state from one state. And there are consequences of taking a certain action in every state, which we call "rewards". The idea of Q-learning is to learn the optimal policy in a Markov Decision Process (MDP). First, where is the term "Q" coming from? "Q" refers to the action-value function that returns the expected return from taking a certain action in the given state and following the given policy thereafter (sum of expected rewards). In other words, it returns the "value" of a certain action in a given state under the given policy. 
 Then, what is the optimal policy in a MDP? It is simply the policy that yields the maximum sum of expected rewards, which is also the outcome of the optimal action-value function (Q-function) given the same state and action. Since the action-value function while following the optimal policy would yield the maximum expected return, Q-learning can learn the optimal policy by learning the action-value function. (will not go into too much details like Bellman optimality and such)

# Deep Q-Learning
## Deep Q-Networks (DQN)
DQN is a reinforcement learning algorithm that combines Q-learning and deep neural networks. DQN approximates the action-value function in the Q-learning framework with a neural network. It utilizes off-policy learning by using Experience Replay from which random samples (could be prioritized) are drawn for neural network training. Experience Replay usually contains a specified number of recent (state, action, next state, reward) "memories". Since DQN learns the action-value function, it is a value-based approach. Once trained, the agent can just take the action incurs the highest action-value given a state. 

# Policy Gradients + Q-Learning (Actor - Critic)
While Policy Gradient algorithms are policy-based since they approximate the policy function direcly, DDPG and SAC try to approximate both the action-value and policy functions, and, thus, called Actor-Critic. The Actor represents the policy, and the Critic represents the action-value function. 

## Deep Deterministic Policy Gradient (DDPG)
One of the main purposes of DDPG is to incorporate continous action space with Deep Q-Learning.  
In DQN, the agent takes the action that incurs the highest action-value by computing the action-value for each action and directly comparing them. However, when the action space is continuous, calculating all the possible action-values can become very tricky (or even impossible) especially if the action space is continous and high dimensional. Then, how does DDPG find the best action? Since the action space is continuous, the Q-function is presumed to be differentialbe w.r.t the action argument, which allows us to perform gradient ascent w.r.t the policy network's parameters (actor network)! Now, how do we optimize the critic network? We use mean-squared Bellman error (MSBE) which tells us how close the critic network is to satisfying the Bellman equation.


## Soft Actor Critic (SAC)
SAC shares the same approach with DDPG in general. However, SAC's central features are entropy regularization and stochastic policy optimization. SAC tries to maximize the tradeoff between expected return and entropy, which is a measure of randomness of the policy. By also focusing on the entropy of the policy, SAC can prevent its policy from prematurely converging to a local optimum. Also, the noise from the stochasticity of the policy brings the similar effect of target policy smoothing. 


# References:
1. DQN Paper
2. DDPG Paper
3. SAC Paper
4. SAC + PER
5. openai 

The state-value function is the expected return starting from state s following the given policy. 



