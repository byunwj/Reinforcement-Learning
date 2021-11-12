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
        
        self.state_size = state_size
        self.action_size = action_size
        self.action_lower_bound = action_lower_bound
        self.action_upper_bound = action_upper_bound

        self.states= tf.compat.v1.placeholder(tf.float32, [None, state_size], 's')
        self.next_states = tf.compat.v1.placeholder(tf.float32, [None, state_size], 's_')
        self.rewards = tf.compat.v1.placeholder(tf.float32, [None, 1], 'r')
        self.target_entropy = tf.constant(-1, dtype=tf.float32)
        #self.alpha = tf.Variable(0.2, dtype=tf.float32)


        with tf.compat.v1.variable_scope('Actor'):
            self.mu, self.std, self.actions, self.log_pi = self.build_actor(self.states, scope='eval', trainable=True) 
            _, _, self.actions_next, self.log_pi_next = self.build_actor(self.next_states, scope='target', trainable= False) 

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
        soft_q_target = min_q_target - self.alpha * self.log_pi_next
        y = tf.stop_gradient(self.rewards + self.config.GAMMA * soft_q_target)

        # calculate the td-errors for the two q-networks
        td_error1 = tf.compat.v1.losses.mean_squared_error(labels = y, predictions=q1)
        td_error2 = tf.compat.v1.losses.mean_squared_error(labels = y, predictions=q2)

        # two separate optimizers for the two q-networks
        self.critic1_optimizer = tf.compat.v1.train.AdamOptimizer(self.config.CRITIC_LR).minimize(td_error1, var_list=self.critic1_params)
        self.critic2_optimizer = tf.compat.v1.train.AdamOptimizer(self.config.CRITIC_LR).minimize(td_error2, var_list=self.critic2_params)
        
        # Get the minimum Q value of the 2 q-networks
        min_q = tf.minimum(q1, q2)
        a_loss = -tf.reduce_mean(min_q - self.alpha * self.log_pi)    # to maximize the min_q - alpha*log_pi
        self.actor_optimizer = tf.compat.v1.train.AdamOptimizer(self.config.ACTOR_LR).minimize(a_loss, var_list=self.actor_params)
        
        self.sess.run(tf.compat.v1.global_variables_initializer())


    # policy sampling will be restricted to -1 to +1 with a tanh function
    def build_actor(self, states, scope, trainable):
        with tf.compat.v1.variable_scope(scope):
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
        indices = np.random.choice(self.config.MEMORY_CAPACITY, size=self.config.BATCH_SIZE)
        bt = self.memory[indices, :]
        bs = bt[:, :self.state_size]
        ba = bt[:, self.state_size: self.state_size + self.action_size]
        br = bt[:, -self.state_size - 1: -self.state_size]
        bs_ = bt[:, -self.state_size:]

        #print('memory buffer', self.memory.shape)
        #print('states shape', bs.shape)
        #print('actions shape', ba.shape)
        #print('rewards shape', br.shape)
        #print('next_states shape', bs_.shape)


        self.sess.run(self.actor_optimizer, {self.states: bs})
        self.sess.run(self.critic1_optimizer, {self.states: bs, self.actions: ba, self.rewards : br, self.next_states: bs_})
        self.sess.run(self.critic2_optimizer, {self.states: bs, self.actions: ba, self.rewards : br, self.next_states: bs_})
        #self.sess.run(self.alpha_optimizer, {self.states: bs})


        # soft target replacement
        self.sess.run(self.soft_replace_actor)
        self.sess.run(self.soft_replace_critic1)
        self.sess.run(self.soft_replace_critic2)
        
        #mu = np.mean(self.sess.run(self.mu, {self.states: bs}))
        #std = np.mean(self.sess.run(self.std, {self.states: bs}))

        


    def remember(self, state, action, reward, next_state):
        # store the unnormalized reward
        self.reward_memory.append(reward)
        self.recent_reward.append(reward)

        if self.config.COUNTER % self.reward_norm_steps == 0: # updates the reward mean and standard deviation every n steps
            self.reward_mean = np.min(self.reward_memory)
            
            print("rewards normalization updated; mean of the last 1000 rewards:{}".format(np.mean(self.recent_reward)))
            print("rewards normalization updated; mean of the last 20000 rewards:{}".format(np.mean(self.reward_memory)))
        
        if self.config.COUNTER > self.config.MEMORY_CAPACITY*0.1:
            if reward != 0:
                reward = -reward/(self.reward_mean + 1e-6)

        transition = np.hstack((state, action, [reward], next_state))
        index = self.config.COUNTER % self.config.MEMORY_CAPACITY  # replace the old memory with new memory
        self.memory[index, :] = transition
        self.config.COUNTER += 1




