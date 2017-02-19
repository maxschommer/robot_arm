# -*- coding: utf-8 -*-
import copy
import gym
import matplotlib.pyplot as plt
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import RMSprop


class DQNAgent:
	def __init__(self, env, mem_size=10000, gamma=0.9, epsilon=0.7, epsilon_decay=0.99, epsilon_min=0.005, learning_rate=0.0003, batch_size=32):
		"""Assign default values to various variables"""
		self.env = env
		self.memory = deque(maxlen=mem_size)
		self.gamma = gamma  # decay rate
		self.epsilon = epsilon  # exploration
		self.epsilon_decay = epsilon_decay
		self.epsilon_min = epsilon_min
		self.learning_rate = learning_rate
		self.batch_size = batch_size
		self._build_model()


	def _build_model(self):
		"""Construct a neural network to serve as the deep Q function"""
		N_IN = self.env.observation_space.shape[0]
		N_OUT = self.env.action_space.n

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
			target_f = np.ones((1,self.env.action_space.n))
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


	def train(self, episodes=1000, verbose=False):
		"""do the machine learning thing"""
		scores = []
		for e in range(episodes):
			state = self.env.reset() # set the environment
			state = np.reshape(state, [1, 4])

			for time_t in range(5000): # start an episode
				if verbose:	self.env.render()	# display
				action = self.act(state) #choose action
				next_state, reward, done, _ = self.env.step(action) # do that action
				next_state = np.reshape(next_state, [1, 4])
				reward = -100 if done else reward
				self.remember(state, action, reward, next_state) # save the result for later analysis
				state = copy.deepcopy(next_state)
				if done: break

			if verbose:
				print("episode: {}/{}, score: {}, memory size: {}, e: {}"
						  .format(e, episodes, time_t,
								  len(self.memory), self.epsilon)) # log some stuff in stdout
			scores.append(time_t)
			if e % 50 == 0:
				self.save("./save/cartpole.h5") # save the current weights
			self.replay(self.batch_size) # train on 32 of the

		if verbose:
			plt.plot(scores)
			plt.ylabel("Time balanced")
			plt.xlabel("Episodes")
			plt.savefig("figure_1.png")
			plt.show()

		return np.mean(scores)


	def __str__(self):
		return "DQNAgent(ms={}, g={}, e={}, ed={}, em={}, lr={}, bs={})".format(
				len(self.memory), self.gamma, self.epsilon, self.epsilon_decay,
				self.epsilon_min, self.learning_rate, self.batch_size)


if __name__ == "__main__":
	env = gym.make('CartPole-v0')
	agent = DQNAgent(env)
	print(agent)
	#agent.load("./save/cartpole-starter.h5")
	score = agent.train(verbose=True)
	print("Final score: "+score)