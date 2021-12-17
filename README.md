Reinforcement Learning
======================

This repository contains a brief explanation and code of the two most well-known approaches of model-free reinforcement learning algorithms, Deep Q-Learning and Policy Gradients (or rather Q-Learning + Policy Gradients). In **very** Simple words, Q-learning aims to learn the reward of a certain action (single deterministic) in a certain state and take a greedy action accordingly while policy gradient method aims to directly learn the best action (could be stochastic) in a certain state.

Note that DQN was implemented using Keras while DDPG and SAC were implemented at the Tensorflow level (i.e. TF v1). Updating the actor network's parameters based on the output of the critic network (to maximize the Q-value, which is the sum of future rewards) could have been implemented using Keras by using tf.GradientTape(), apply_gradients, and such. However, the speed was noticeably slower compared to when implemented with TF v1.


# Q-Learning
While in Markov Process (MP), the transition probability dictates the next state from a certain state, in MDP, the action dictates the next state from one state. And there are consequences of taking a certain action in every state, which we call "rewards". The idea of Q-learning is to learn the optimal policy in a Markov Decision Process (MDP). First, where is the term "Q" coming from? "Q" refers to the action-value function that returns the expected return from taking a certain action in the given state and following the given policy thereafter (sum of expected rewards). In other words, it returns the "value" of a certain action in a given state under the given policy. 
Then, what is the optimal policy in a MDP? It is simply the policy that yields the maximum sum of expected rewards, which is also the outcome of the optimal action-value function (Q-function) given the same state and action. Since the action-value function would yield the maximum expected return if the agent follows the optimal policy, Q-learning can learn the optimal policy by learning the action-value function. (will not go into too much details of the Bellman optimality and such)

# Deep Q-Learning
## Deep Q-Networks (DQN)
DQN is a reinforcement learning algorithm that combines Q-learning and deep neural networks. DQN approximates the action-value function in the Q-learning framework with a neural network. It utilizes off-policy learning by using Experience Replay from which random samples (could be prioritized) are drawn for neural network training. Experience Replay usually contains a specified number of recent (state, action, next state, reward) "memories". Since DQN learns(approximates) the action-value function, it is a value-based approach. Once trained, the agent can just take the action that incurs the highest action-value under a given state. 

# Policy Gradients + Q-Learning (Actor - Critic)
Policy Gradient, on the other hand, is a policy-based approach since it approximates the policy function direcly. Furthermore, DDPG and SAC try to approximate both the action-value and policy functions with neural networks, thus called Actor-Critic. The Actor represents the policy, and the Critic represents the action-value function. 

## Deep Deterministic Policy Gradient (DDPG)
One of the main purposes of DDPG is to incorporate continous action space with Deep Q-Learning.  
In DQN, the agent takes the action that incurs the highest action-value by computing the action-value for each action and directly comparing them. However, when the action space is continuous, calculating all the possible action-values can become very tricky (or even impossible) especially if the action space is continous and high dimensional. Then, how does DDPG find the best action? Since the action space is continuous, the Q-function is presumed to be differentialbe w.r.t the action argument, which allows us to perform gradient ascent w.r.t the policy network's parameters (actor network)! Now, how do we optimize the critic network? We use mean-squared Bellman error (MSBE) which tells us how close the output of the critic network is to the output of the optimal action-value function, which is equal to satisfying the Bellman optimality.
 

## Soft Actor Critic (SAC)
SAC shares the same approach with DDPG in general. However, SAC's central features are entropy regularization and stochastic policy optimization. SAC tries to maximize the tradeoff between expected return and entropy, which is a measure of randomness of the stochastic policy. The higher the entropy, the more evenly distributed the probabilities of possible actions. Thus, by including the entropy term in the target Q (optimal Q), SAC forces its policy to explore more actions
which can prevent its policy from prematurely converging to a local optimum. Also, the stochastic policy of SAC learns the mean and standard deviation of the action distribution. When the agent needs to take an action, a single action is then sampled from the distribution. The sampling process inherently adds noise to the policy which brings the similar effect of target policy smoothing (a technique used in Twin Delayed DDPG (TD3) algorithm). 

# Implementation
The DQN implementation solves the CartPole problem from OPENAI Gym environment, and the DDPG and SAC implementations solve the Pendulum problem from OPENAI Gym environment. I've generated a return graph for each implementation to see how the return of each episode changes overtime as the agent goes through training. While DQN and DDPG has only one 'version', SAC was implemented with a couple of add-ons such as reward normalization and Prioritized Experience Replay (PER) which were compared with the version without any add-ons. 

The following comparisons were made:
1. DDPG vs. SAC
2. SAC  vs. SAC with reward normalization
3. SAC  vs. SAC with PER
Comparisons were made based on the return graphs and the returns of the last 200 episodes.  
Comparing DDPG and SAC's return graphs, we can see that the return graph of SAC is slightly more stable than that of DDPG as there are no returns less than -500 after episode 100 (which seems to be just slightly after the return starts to converge) for SAC while there is a few returns that spike under -500 for DDPG. Comparing SAC and SAC with reward normalization, there seems to be no advantage in applying reward normalization. In fact, SAC **without** reward normalization seems to converge earlier (before episode 100 while SAC with reward normalization converges around episode 150-200), and it seems that the stability after converging is not necessarily improved by addding reward normalization at least in this example as the mean return of the last 100 episodes is slightly lower in the case with reward normalization. It is also surprising to see that adding PER does not necessarily yield better training result as the mean return of the last 100 episodes is again slightly lower in the case with PER compared to the benchmark SAC.  
  
  
<figure>
<figcaption> 
<b>DDPG</b>
</figcaption>
<img src="https://github.com/byunwj/Reinforcement-Learning/blob/main/Deep%20Deterministic%20Policy%20Gradient%20(DDPG)/Return%20Graph%20DDPG.png?raw=true" width="550px" height="400px" title="px(픽셀) 크기 설정" alt="DDPG">
</figure>
</br>
  
<figure>
<figcaption> 
<b>Benchmark SAC</b>
</figcaption>
<img src="https://github.com/byunwj/Reinforcement-Learning/blob/main/Soft%20Actor%20Critic%20(SAC)/Return%20Graph_Reward%20Normalization_False%20%26%20PER_False.png?raw=true" width="550px" height="400px" title="px(픽셀) 크기 설정" alt="Benchmark SAC">
</figure>
</br>
  
<figure>
<figcaption> 
<b>SAC with Reward Normalization</b>
</figcaption>
<img src="https://github.com/byunwj/Reinforcement-Learning/blob/main/Soft%20Actor%20Critic%20(SAC)/Return%20Graph_Reward%20Normalization_True%20%26%20PER_False.png?raw=true" width="550px" height="400px" title="px(픽셀) 크기 설정" alt="SAC with Reward Normalization">
</figure>
</br>
  
<figure>
<figcaption>
<b>SAC with Prioritized Experience Replay</b>
</figcaption>
<img src="https://github.com/byunwj/Reinforcement-Learning/blob/main/Soft%20Actor%20Critic%20(SAC)/Return%20Graph_Reward%20Normalization_False%20%26%20PER_True.png?raw=true" width="550px" height="400px" title="px(픽셀) 크기 설정" alt="SAC with Prioritized Experience Replay">
</figure>
</br>


# References:
1. [DQN](https://arxiv.org/abs/1312.5602)
2. [DDPG](https://arxiv.org/abs/1509.02971)
3. [SAC](https://arxiv.org/abs/1801.01290)
4. [Prioritized Experience Replay (PER)](https://arxiv.org/abs/1511.05952)
5. [openai DDPG](https://spinningup.openai.com/en/latest/algorithms/ddpg.html)
6. [openai SAC](https://spinningup.openai.com/en/latest/algorithms/sac.html) 