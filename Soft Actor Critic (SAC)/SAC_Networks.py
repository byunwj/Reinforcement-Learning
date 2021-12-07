import tensorflow as tf
import numpy as np
from collections import deque
import random


class SAC(object):
    def __init__(self, sess, config, state_size, action_size, action_lower_bound, action_upper_bound, alpha):
        # initialize the parameters and the model
        self.eps = 1e-6
        self.alpha = alpha

        self.config = config
        self.sess = sess 
        self.memory = np.zeros((self.config.MEMORY_CAPACITY, state_size * 2 + action_size + 1), dtype=np.float32)
        self.reward_memory = deque(maxlen =  self.config.MEMORY_CAPACITY)
        self.recent_reward = deque(maxlen =  1000)
        self.reward_norm_steps  = 200
        self.reward_mean        = 1
        self.iteration = 0
        
        # for Prioritized Experience Replay (PER)
        self.priorities = np.zeros((self.config.MEMORY_CAPACITY,), dtype=np.float32)
        self.beta_start = 0.4
        self.beta_frames = 100000
        self.frame = 1 #for beta calculation
        self.pos = 0
        self.per_alpha = 0.6

        self.state_size = state_size
        self.action_size = action_size
        self.action_lower_bound = action_lower_bound
        self.action_upper_bound = action_upper_bound

        self.states= tf.compat.v1.placeholder(tf.float32, [None, state_size], 's')
        self.next_states = tf.compat.v1.placeholder(tf.float32, [None, state_size], 's_')
        self.rewards = tf.compat.v1.placeholder(tf.float32, [None, 1], 'r')
        self.target_entropy = tf.constant(-1, dtype=tf.float32)
        self.weights = tf.compat.v1.placeholder(tf.float32, [None, 1], 'Weights')

        #self.alpha = tf.Variable(0.2, dtype=tf.float32)


        with tf.compat.v1.variable_scope('Actor'):
            self.mu, self.std, self.actions, self.log_pi = self.build_actor(self.states, scope='eval', trainable=True) 
            _, _, self.actions_next, self.log_pi_next = self.build_actor(self.next_states, scope='target', trainable= False) 
            #_, _, self.actions_next, self.log_pi_next = self.build_actor(self.next_states, scope='eval', trainable= True) 
            
        with tf.compat.v1.variable_scope('Critic'):
            q1 = self.build_critic(self.states, self.actions, scope='eval1', trainable=True)
            q2 = self.build_critic(self.states, self.actions, scope='eval2', trainable=True)
            
            q1_target = self.build_critic(self.next_states, self.actions_next, scope='target1', trainable=False)
            q2_target = self.build_critic(self.next_states, self.actions_next, scope='target2', trainable=False)
        

        # networks parameters
        self.actor_params = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, scope='Actor/eval')
        self.actor_t_params = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, scope='Actor/target')
        
        self.critic1_params = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, scope='Critic/eval1')
        self.critic1_t_params = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, scope='Critic/target1')

        self.critic2_params = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, scope='Critic/eval2')
        self.critic2_t_params = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, scope='Critic/target2')


        # target net replacement (two targets for the two q-networks)
        self.soft_replace_critic1 = [tf.compat.v1.assign(t, (1 - self.config.TAU) * t + self.config.TAU * e) for t, e in zip(self.critic1_t_params, self.critic1_params)]
        self.soft_replace_critic2 = [tf.compat.v1.assign(t, (1 - self.config.TAU) * t + self.config.TAU * e) for t, e in zip(self.critic2_t_params, self.critic2_params)]
        self.soft_replace_actor = [tf.compat.v1.assign(t, (1 - self.config.TAU) * t + self.config.TAU * e) for t, e in zip(self.actor_t_params, self.actor_params)]

        
        # Apply the clipped double Q trick --> Get the minimum Q value of the 2 target q-networks
        min_q_target = tf.minimum(q1_target, q2_target)
        
        # Add the entropy term to get soft Q target
        soft_q_target = min_q_target - self.alpha * tf.stop_gradient(self.log_pi_next)
        y = self.rewards + self.config.GAMMA * soft_q_target

        # calculate the td-errors for the two q-networks
        td_error1 = tf.compat.v1.losses.mean_squared_error(labels = y, predictions=q1)
        td_error2 = tf.compat.v1.losses.mean_squared_error(labels = y, predictions=q2)

        self.td_error1_weighted = td_error1*self.weights
        self.td_error2_weighted = td_error2*self.weights
    
        self.updated_priorities = abs( ( (y-q1) + (y-q2) )/2 + 1e-5  )

        # two separate optimizers for the two q-networks
        self.critic1_optimizer = tf.compat.v1.train.AdamOptimizer(self.config.CRITIC_LR).minimize(self.td_error1_weighted, var_list=self.critic1_params)
        self.critic2_optimizer = tf.compat.v1.train.AdamOptimizer(self.config.CRITIC_LR).minimize(self.td_error2_weighted, var_list=self.critic2_params)
        #self.critic1_optimizer = tf.compat.v1.train.AdamOptimizer(self.config.CRITIC_LR).minimize(td_error1, var_list=self.critic1_params)
        #self.critic2_optimizer = tf.compat.v1.train.AdamOptimizer(self.config.CRITIC_LR).minimize(td_error2, var_list=self.critic2_params)
        
        # Get the minimum Q value of the 2 q-networks
        min_q = tf.minimum(q1, q2)
        a_loss = -tf.reduce_mean(min_q - self.alpha * self.log_pi)    # to maximize the min_q - alpha*log_pi
        self.actor_optimizer = tf.compat.v1.train.AdamOptimizer(self.config.ACTOR_LR).minimize(a_loss, var_list=self.actor_params)
        
        self.sess.run(tf.compat.v1.global_variables_initializer())


    # policy sampling will be restricted to -1 to +1 with a tanh function
    def build_actor(self, states, scope, trainable):
        with tf.compat.v1.variable_scope(scope, reuse = tf.compat.v1.AUTO_REUSE):
            n_l1 = 256
            prob = tf.compat.v1.layers.dense(states, n_l1, trainable=trainable)
            prob = tf.nn.relu(prob)
            prob = tf.compat.v1.layers.dense(prob, n_l1, trainable=trainable)
            prob = tf.nn.relu(prob)

            mu = tf.compat.v1.layers.dense(prob, 1, trainable=trainable) 
            log_std = tf.compat.v1.layers.dense(prob, 1, trainable=trainable)
            log_std = tf.clip_by_value(log_std, -20, 2) # -20 and 2 are just consequences of experiments of the authors

            # Standard deviation is bounded by a constraint of being non-negative --> we produce log_std as output which can be [-inf, inf]
            std = tf.exp(log_std)

            actions, log_pi = self.sample_action(mu, std)
            
            return mu, std, actions, log_pi


    def sample_action(self, mu, std):
        # Use re-parameterization trick to deterministically sample action from
        # the policy network. First, sample from a Normal distribution of
        # sample size as the action and multiply it with stdev
        dist = tf.compat.v1.distributions.Normal(mu, std)
        actions_ = dist.sample()

        # Calculate the log probability
        log_pi = dist.log_prob(actions_)

        # Apply the tanh squashing to keep the gaussian bounded in (-1,1) 
        actions = tf.tanh(actions_)

        # Change log probability to account for tanh squashing as mentioned in
        # Appendix C of the paper (pg. 12 of the ppt used in seminar)
        log_pi -= tf.reduce_sum(tf.math.log(1 - actions**2 + self.eps), axis=1, keepdims=True)
        
        # then multiply by 2 --> bounded in (-2,2)
        actions *= self.action_upper_bound

        return actions, log_pi



    # need to make two q-networks (clippled double q-learning)
    def build_critic(self, states, actions, scope, trainable):
        with tf.compat.v1.variable_scope(scope):
            n_l1 = 256
            q = tf.keras.layers.Concatenate()([states, actions])    
            q = tf.compat.v1.layers.dense(q, n_l1, trainable=trainable)
            q = tf.nn.relu(q)
            q = tf.compat.v1.layers.dense(q, n_l1, trainable=trainable)
            q = tf.nn.relu(q)
            q = tf.compat.v1.layers.dense(q, 1, trainable = trainable)
    
            return q  # Q(s,a)


    def act(self, states):
        return self.sess.run(self.actions, {self.states: states[np.newaxis, :]})[0]
    
        
    def train(self):
        if self.iteration == 0:
            print('\t\t\t\t\t\t\t\t----------- SAC TRAINING HAS STARTED -----------\t\t\t\t\t\t\t\t')

        # random sampling
        #indices = np.random.choice(self.config.MEMORY_CAPACITY, size=self.config.BATCH_SIZE)
        #bt = self.memory[indices, :]
        #bs = bt[:, :self.state_size]
        #ba = bt[:, self.state_size: self.state_size + self.action_size]
        #br = bt[:, -self.state_size - 1: -self.state_size]
        #bs_ = bt[:, -self.state_size:]

        # prioritized sampling
        indices, weights = self.get_indices()
        weights = np.expand_dims(weights, axis=1) # so that it can be a tensor input with shape (128, 1)
        bt = self.memory[indices, :]
        bs = bt[:, :self.state_size]
        ba = bt[:, self.state_size: self.state_size + self.action_size]
        br = bt[:, -self.state_size - 1: -self.state_size]
        bs_ = bt[:, -self.state_size:]

        _,_,_, updated_priorities = \
            self.sess.run([self.actor_optimizer, self.critic1_optimizer, self.critic2_optimizer, self.updated_priorities],
                          {self.states: bs, self.actions: ba, self.rewards : br, \
                           self.next_states: bs_, self.weights: weights})

        #print("old priorities: \n", self.priorities[indices])
        self.update_priorities(indices, updated_priorities.squeeze(1))
        #print("new priorities: \n", self.priorities[indices])
        #import sys; sys.exit()

        # soft target replacement
        if self.iteration % 100 == 0:
            self.sess.run([self.soft_replace_critic1, self.soft_replace_critic2, self.soft_replace_actor])
        
        #if self.iteration % 200 == 0:
            #print(updated_priorities)
        self.iteration += 1
        #mu = np.mean(self.sess.run(self.mu, {self.states: bs}))
        #std = np.mean(self.sess.run(self.std, {self.states: bs}))

        


    def remember(self, state, action, reward, next_state):
        # store the unnormalized reward
        self.reward_memory.append(reward)
        self.recent_reward.append(reward)

        max_prio = self.priorities.max() if self.config.COUNTER > 0 else 1.0

        if self.config.COUNTER % self.reward_norm_steps == 0: # updates the reward mean and standard deviation every n steps
            self.reward_mean = np.mean(self.reward_memory)
            
            print("rewards normalization updated; mean of the last 1000 rewards:{}".format(np.mean(self.recent_reward)))
            print("rewards normalization updated; mean of the last 20000 rewards:{}".format(np.mean(self.reward_memory)))
        
        if self.config.COUNTER > self.config.MEMORY_CAPACITY*0.1:
            if reward != 0:
                reward = (reward - self.reward_mean)/( np.std(self.reward_memory) + 1e-6)

        # insert the transition into the replay buffer at the end
        transition = np.hstack((state, action, [reward], next_state))
        index = self.config.COUNTER % self.config.MEMORY_CAPACITY  # replace the old memory with new memory
        self.memory[index, :] = transition
        
        # insert the priority of the transition at the end
        if self.pos < self.config.MEMORY_CAPACITY:
            self.priorities[self.pos] = max_prio
            self.pos = self.pos + 1
        else:
            self.priorities[:-1] = self.priorities[1:]
            self.priorities[-1]  = max_prio
        
        self.config.COUNTER += 1




    def beta_by_frame(self, frame_idx):
        return min(1.0, self.beta_start + frame_idx * (1.0 - self.beta_start) / self.beta_frames)
    
    def get_indices(self):
        N = len(self.memory) if self.config.COUNTER >= len(self.memory) else self.config.COUNTER
        if N == self.config.MEMORY_CAPACITY: # if the memory is full, use the whole priorities
        #if self.config.COUNTER >= len(self.memory):
            prios = self.priorities
        else:
            prios = self.priorities[:self.pos]

        # calc P = p^a/sum(p^a)
        probs  = prios ** self.per_alpha
        P = probs/probs.sum() # so that the probabilities add up to 1

        #gets the indices depending on the probability p
        indices = np.random.choice(N, self.config.BATCH_SIZE, p=P) 

        beta = self.beta_by_frame(self.frame)
        self.frame+=1

        #Compute importance-sampling weight
        weights  = (N * P[indices]) ** (-beta)
        # normalize weights
        weights /= weights.max() 
        weights  = np.array(weights, dtype=np.float32) 

        return indices, weights

    def update_priorities(self, batch_indices, batch_priorities):
        for idx, prio in zip(batch_indices, batch_priorities):
            self.priorities[idx] = abs(prio) 



