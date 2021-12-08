import tensorflow as tf
import numpy as np
from collections import deque
import random


class DDPG(object):
    def __init__(self, sess, config, state_size, action_size, action_lower_bound, action_upper_bound):
        # initialize the parameters and the model
        self.config = config
        self.sess = sess
        self.memory = np.zeros((self.config.MEMORY_CAPACITY, state_size * 2 + action_size + 1), dtype=np.float32)
        
        self.state_size = state_size
        self.action_size = action_size
        self.action_lower_bound = action_lower_bound
        self.action_upper_bound = action_upper_bound

        self.states= tf.compat.v1.placeholder(tf.float32, [None, state_size], 's')
        self.next_states = tf.compat.v1.placeholder(tf.float32, [None, state_size], 's_')
        self.rewards = tf.compat.v1.placeholder(tf.float32, [None, 1], 'r')

        with tf.compat.v1.variable_scope('Actor'):
            self.actions = self.build_actor(self.states, scope='eval', trainable=True)
            actions_target = self.build_actor(self.next_states, scope='target', trainable=False)

        with tf.compat.v1.variable_scope('Critic'):
            # assign self.a = a in memory when calculating q for td_error,
            # otherwise the self.a is from Actor when updating Actor
            q = self.build_critic(self.states, self.actions, scope='eval', trainable=True)
            q_target = self.build_critic(self.next_states, actions_target, scope='target', trainable=False)

        # networks parameters
        self.actor_params = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, scope='Actor/eval')
        self.actor_t_params = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, scope='Actor/target')
        self.critic_params = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, scope='Critic/eval')
        self.critic_t_params = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, scope='Critic/target')

        # target net replacement
        self.soft_replace = [tf.compat.v1.assign(t, (1 - self.config.TAU) * t + self.config.TAU * e)
                             for t, e in zip(self.actor_t_params + self.critic_t_params, self.actor_params + self.critic_params)]

        q_target = self.rewards + self.config.GAMMA * q_target
        # in the feed_dic for the td_error, the self.a should change to actions in memory
        td_error = tf.compat.v1.losses.mean_squared_error(labels=q_target, predictions=q)
        self.critic_optimizer = tf.compat.v1.train.AdamOptimizer(self.config.CRITIC_LR).minimize(td_error, var_list=self.critic_params)

        a_loss = -tf.reduce_mean(q)    # maximize the q
        self.actor_optimizer = tf.compat.v1.train.AdamOptimizer(self.config.ACTOR_LR).minimize(a_loss, var_list=self.actor_params)

        self.sess.run(tf.compat.v1.global_variables_initializer())


    def build_actor(self, states, scope, trainable):
        with tf.compat.v1.variable_scope(scope):
            out = tf.compat.v1.layers.dense(states, 128, trainable=trainable)
            out = tf.nn.relu(out)
            out = tf.compat.v1.layers.dense(out, 64, trainable=trainable)
            out = tf.nn.relu(out)
            out = tf.compat.v1.layers.dense(out, 1, trainable=trainable) 
            out = tf.tanh(out)*2
            return out

    def build_critic(self, states, actions, scope, trainable):
        with tf.compat.v1.variable_scope(scope):
            q = tf.keras.layers.Concatenate()([states, actions])    
            q = tf.compat.v1.layers.dense(q, 128, trainable=trainable)
            q = tf.nn.relu(q)
            q = tf.compat.v1.layers.dense(q, 64, trainable=trainable)
            q = tf.nn.relu(q)
            q = tf.compat.v1.layers.dense(q, 1, trainable = trainable)
            return q  # Q(s,a)

    def act(self, states):
        return np.clip(np.random.normal(self.sess.run(self.actions, {self.states: states[np.newaxis, :]})[0], self.config.STAND_DEV), self.action_lower_bound, self.action_upper_bound)
        
    def train(self):
        self.config.STAND_DEV *= .9995

        # soft target replacement
        self.sess.run(self.soft_replace)

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
        self.sess.run(self.critic_optimizer, {self.states: bs, self.actions: ba, self.rewards : br, self.next_states: bs_})

    def remember(self, state, action, reward, next_state):
        transition = np.hstack((state, action, [reward], next_state))
        index = self.config.COUNTER % self.config.MEMORY_CAPACITY  # replace the old memory with new memory
        self.memory[index, :] = transition
        self.config.COUNTER += 1
