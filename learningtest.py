import gym
from keras import models
from keras import layers

import numpy as np
import random as rng
import time

episodes = 1000
N = 100
epsilon0 = .2
gamma = 0.7

env = gym.make('CartPole-v0')

network = models.Sequential() # network models a function Q(s), which returns a vector

network.add(layers.Dense(4, activation='relu', input_shape=(1,4)))#, dim_sorting='th'))
network.add(layers.Dense(env.action_space.n, activation='softmax'))

network.compile(loss='categorical_crossentropy',
				optimizer='adam',
				metrics=['accuracy'])

epsilon = epsilon0
inputs = []
targets = []

for i_episode in range(episodes):
	state = env.reset()
	t = 0
	done = False

	while not done:
		a = time.time()
		env.render()

		b = time.time()
		if rng.random() < epsilon:
			action = env.action_space.sample()
		else:
			Q_s1 = network.predict(np.array([state]))
			action = np.argmax(Q_s1)

		c = time.time()
		inputs.append(state)
		state, reward, done, info = env.step(action)
		Q_s2 = network.predict(np.array([state]))

		d = time.time()
		target_vector = np.zeros((env.action_space.n,))
		target_vector[action] = reward + gamma*np.max(Q_s2)
		targets.append(target_vector)

		e = time.time()
		batch_size = min(len(inputs), N)
		network.train_on_batch(
				np.array([inputs]),
				np.array([targets]))

		f = time.time()
		t += 1
		print("{:04.2f},{:04.2f},{:04.2f},{:04.2f},{:04.2f}".format(b-a,c-b,d-c,e-d,f-e))

	print("Episode finished after {} timesteps".format(t+1))
	epsilon = .2
	done = False
