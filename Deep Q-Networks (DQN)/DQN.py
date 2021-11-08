import os
import random
import gym
import numpy as np
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from collections import deque

class DQNAgent():

    def __init__(self, hyper_params):
        
        self.env = gym.make("CartPole-v1")
        self.state_space = self.env.observation_space.shape[0] #4; location, angle, etc. 
        self.action_space = self.env.action_space.n #2, 0 or 1

        self.experience = deque(maxlen=2000)

        self.gamma = hyper_params["gamma"] #discount rate
        self.epsilon = hyper_params["epsilon"] #exploration rate
        self.episodes = hyper_params ["episodes"] 
        self.min_epsilon = hyper_params["min_epsilon"]
        self.epsilon_decay = hyper_params["epsilon_decay"]
        self.fitting_start = hyper_params["fitting_start"]
        self.learning_rate = hyper_params["learning_rate"] 
        self.batch_size = hyper_params["batch_size"]

        self.model = self.policy_model()
        self.target_model = self.policy_model()

    def policy_model(self):
        model = Sequential()
        model.add(Dense(256, input_dim = self.state_space, activation = 'relu'))
        model.add(Dense(128, activation= 'relu'))
        model.add(Dense(64, activation= 'relu'))
        model.add(Dense(self.action_space, activation='linear'))

        model.compile(loss='mse', optimizer = Adam(lr=self.learning_rate), metrics=["accuracy"]) #not sure about the loss function
        return model
    

    def agent_step(self, state):
        if (np.random.random() < self.epsilon):
            return random.randrange(self.action_space)     #exploration
        else: 
            return np.argmax(self.model.predict(state))    #exploitation


    def add_experience(self, state, action, reward, next_state, done):
        self.experience.append((state, action, reward, next_state, done))


    def model_fitting(self): #training the model
        sample_exp = random.sample(self.experience, self.batch_size)
        batch = list(zip(*sample_exp))
        state = np.reshape(list(batch[0]), [len(sample_exp), self.state_space])
        next_state = np.reshape(list(batch[3]), [len(sample_exp), self.state_space])
        actions = list(batch[1]); reward = list(batch[2]); done = list(batch[4])
        
        current_q = self.model.predict(state) #initializing with current weights
        next_q = self.target_model.predict(next_state) #using the target network

        #update the current_q to be closer to the optimal q values using bellman
        for j in range(len(sample_exp)):
            action = actions[j]
            if done[j]: 
                current_q[j][action] = reward[j] #there is no target_q value
            else:
                current_q[j][action] = reward[j] + self.gamma*(np.amax(next_q[j]))
        
        self.model.fit(state, current_q, batch_size = 32, verbose = 0, shuffle = True) #updating the weights
       

    
    def model_main(self):
        step = 1
        for ep in range(self.episodes):
            state = self.env.reset()
            state = np.reshape(state, [1, self.state_space]) #to make sure the input into the network is (4,)
            score = 0
            done = False
            
            while not done:
                self.env.render()
                action = self.agent_step(state)
                next_state, reward, done, _ = self.env.step(action)
                next_state = np.reshape(next_state, [1, self.state_space]) #to make sure the input into the network is (4,)
                if not done:
                    self.add_experience(state, action, reward, next_state, done)
                    score += 1
                    state = next_state
                    step += 1
                else:
                    reward = -100
                    self.add_experience(state, action, reward, next_state, done)
                    score += 1
                    print("episodes: " + str(ep+1) + " score: " + str(score) + " epsilon: " + str(self.epsilon))
                    if score == 500:
                        print("saving the model")
                        self.save("carpole-dqn-final.h5")
                        return
              
                if (len(self.experience) > self.fitting_start): #check if there's enough "data" to start fitting the model
                    self.model_fitting()
                    if (self.epsilon > self.min_epsilon):
                        self.epsilon *= self.epsilon_decay   

                if step % 100 == 0:
                    print("update the target network")
                    self.target_model.set_weights(self.model.get_weights())             
                

    def model_test(self):
        self.load_m("carpole-dqn-final.h5")
        for ep in range(self.episodes):
            state = self.env.reset()
            state = np.reshape(state, [1, self.state_space])
            done = False
            score = 0
    
            while not done:
                self.env.render()
                action = np.argmax(self.model.predict(state))
                next_state, reward, done, _ = self.env.step(action)
                state = np.reshape(next_state, [1, self.state_space])
                score += 1
                if done:
                    print("episodes: " + str(ep+1) + " score: " + str(score))
                    break

    
    def save(self, model_name):
        self.model.save(model_name)

    def load_m(self, model_name):
        self.model = load_model(model_name)

                
                



if __name__ == "__main__":
    hyper_params = {"gamma": 0.99, "epsilon": 1.0, "episodes": 2000, "min_epsilon": 0.001, "epsilon_decay":0.999, "fitting_start": 1000 ,"learning_rate": 0.001, "batch_size": 128}
    agent = DQNAgent(hyper_params)
    agent.model_main()
    #agent.model_test() #run after agent.model_run()

