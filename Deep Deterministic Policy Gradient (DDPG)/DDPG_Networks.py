# sean sungil kim

import tensorflow as tf
import numpy as np
from collections import deque
import random


class DDPG(object):
    def __init__(self, sess, config, state_size, action_size, action_lower_bound, action_upper_bound):
        # initialize the parameters and the model
        self.config = config
        self.sess = sess
        if self.config.MEM_SETTING == 'np':
            self.memory = np.zeros((self.config.MEMORY_CAPACITY, state_size * 2 + action_size + 1), dtype=np.float32)
        else:
            self.memory = deque(maxlen = self.config.MEMORY_CAPACITY)

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
            net = tf.compat.v1.layers.dense(states, 30, activation=tf.nn.relu, name='l1', trainable=trainable)
            a = tf.compat.v1.layers.dense(net, self.action_size, activation=tf.nn.tanh, name='a', trainable=trainable)
            return tf.multiply(a, self.action_upper_bound, name='scaled_a')

    def build_critic(self, states, actions, scope, trainable):
        with tf.compat.v1.variable_scope(scope):
            n_l1 = 30
            w1_s = tf.compat.v1.get_variable('w1_s', [self.state_size, n_l1], trainable=trainable)
            w1_a = tf.compat.v1.get_variable('w1_a', [self.action_size, n_l1], trainable=trainable)
            b1 = tf.compat.v1.get_variable('b1', [1, n_l1], trainable=trainable)
            net = tf.nn.relu(tf.matmul(states, w1_s) + tf.matmul(actions, w1_a) + b1)
            return tf.compat.v1.layers.dense(net, 1, trainable=trainable)  # Q(s,a)

    def act(self, states):
        return np.clip(np.random.normal(self.sess.run(self.actions, {self.states: states[np.newaxis, :]})[0], self.config.STAND_DEV), self.action_lower_bound, self.action_upper_bound)

    def train(self):
        if self.config.MEM_SETTING == 'np':
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

        else:
            # decrease the epsilon value
            #self.config.EPSILON *= self.config.EPSILON_DECAY
            self.config.STAND_DEV *= self.config.EPSILON_DECAY

            field_names = ['state', 'action', 'reward', 'next_state']
            batch_data = {}

            # randomly sample from the replay experience que
            replay_batch = random.sample(self.memory, self.config.BATCH_SIZE)
            for i in range(len(field_names)):
                batch_data[field_names[i]] = [data for data in list(zip(*replay_batch))[i]]
            
            #print('\nmemory buffer', np.array(self.memory).shape)
            #print('states shape', np.array(batch_data['state']).shape)
            #print('actions shape', np.array(batch_data['action']).shape)
            #print('rewards shape', np.array(batch_data['reward']).shape)
            #print('next_states shape', np.array(batch_data['next_state']).shape)
          
            self.sess.run(self.actor_optimizer, {self.states: np.array(batch_data['state'])})
            self.sess.run(self.critic_optimizer, {self.states: np.array(batch_data['state']),
                                                self.actions: np.array(batch_data['action']),
                                                self.rewards: np.array(batch_data['reward']),
                                                self.next_states: np.array(batch_data['next_state'])})

    def remember(self, state, action, reward, next_state):
        if self.config.MEM_SETTING == 'np':
            transition = np.hstack((state, action, [reward], next_state))
            index = self.config.COUNTER % self.config.MEMORY_CAPACITY  # replace the old memory with new memory
            self.memory[index, :] = transition
            self.config.COUNTER += 1

        else:
            # add 1 to the pointer
            self.config.COUNTER += 1

            # store in the replay experience queue
            self.memory.append((state, action, [reward], next_state))
