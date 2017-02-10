# -*- coding: utf-8 -*-
import copy
import gym
import matplotlib.pyplot as plt
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import RMSprop

episodes = 1000


class DQNAgent:
    def __init__(self, env):
        """Assign default values to various variables"""
        self.env = env
        self.memory = deque(maxlen=10000)
        self.gamma = 0.9  # decay rate
        self.epsilon = 0.7  # exploration
        self.epsilon_decay = .99
        self.epsilon_min = 0.005
        self.learning_rate = 0.0003
        self._build_model()


    def _build_model(self):
        """Construct a neural network to serve as the deep Q function"""
        N_IN = env.observation_space.shape[0]
        N_OUT = env.action_space.n

        model = Sequential()
        model.add(Dense(64, input_dim=N_IN, activation='tanh', init='he_uniform'))
        model.add(Dense(128, activation='tanh', init='he_uniform'))
        model.add(Dense(128, activation='tanh', init='he_uniform'))
        model.add(Dense(N_OUT, activation='linear', init='he_uniform'))

        model.compile(loss='mse',
                      optimizer=RMSprop(lr=self.learning_rate))
        self.model = model


    def remember(self, state, action, reward, next_state):
        """Append an event tuple to the memory"""
        self.memory.append((state, action, reward, next_state))


    def act(self, state):
        """Choose the best action given the state and current Q model"""
        if np.random.rand() <= self.epsilon:
            return self.env.action_space.sample()
        else:
            act_values = self.model.predict(state)
            return np.argmax(act_values[0])  # returns action


    def replay(self, batch_size):
        """train on a random sample of events from memory"""
        size = min(batch_size, len(self.memory))
        batches = np.random.choice(len(self.memory), size)

        for i in batches:
            state, action, reward, next_state = self.memory[i]
            target = reward + self.gamma * \
                        np.amax(self.model.predict(next_state)[0])
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, nb_epoch=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


    def load(self, name):
        """load a set of weights from memory"""
        self.model.load_weights(name)


    def save(self, name):
        """save the current weights to memory"""
        self.model.save_weights(name)


if __name__ == "__main__":
    env = gym.make('CartPole-v0')
    agent = DQNAgent(env)
    #agent.load("./save/cartpole-starter.h5")

    scores = []
    for e in range(episodes):
        state = env.reset() # set the environment
        state = np.reshape(state, [1, 4])

        for time_t in range(5000): # start an episode
            env.render()    # display
            action = agent.act(state) #choose action
            next_state, reward, done, _ = env.step(action) # do that action
            next_state = np.reshape(next_state, [1, 4])
            reward = -100 if done else reward
            agent.remember(state, action, reward, next_state) # save the result for later analysis
            state = copy.deepcopy(next_state)
            if done: break

        print("episode: {}/{}, score: {}, memory size: {}, e: {}"
                  .format(e, episodes, time_t,
                          len(agent.memory), agent.epsilon)) # log some stuff in stdout
        scores.append(time_t)
        if e % 50 == 0:
            agent.save("./save/cartpole.h5") # save the current weights
        agent.replay(32) # train on 32 of the

    plt.plot(scores)
    plt.ylabel("Time balanced")
    plt.xlabel("Episodes")
    plt.show()
