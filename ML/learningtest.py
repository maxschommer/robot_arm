import gym
from keras import models
from keras import layers
from keras import optimizers

import numpy as np
import random as rng
import time

episodes = 1000
N = 100
epsilon0 = 1
gamma = 0.7

env = gym.make('CartPole-v0')

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import RMSprop
model = Sequential()
model.add(Dense(64, input_dim=4, activation='tanh', init='he_uniform'))
model.add(Dense(128, activation='tanh', init='he_uniform'))
model.add(Dense(128, activation='tanh', init='he_uniform'))
model.add(Dense(2, activation='linear', init='he_uniform'))
model.compile(loss='mse',
              optimizer=RMSprop(lr=0.0001))


network = model

# network.compile(loss='categorical_crossentropy',
# 				optimizer='adam',
# 				metrics=['accuracy'])

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
		print('I feed it',state.shape)
		Q_s2 = network.predict(state)

		d = time.time()
		target_vector = np.zeros((env.action_space.n,))
		target_vector[action] = reward + gamma*np.max(Q_s2)
		targets.append(target_vector)

		e = time.time()
		batch_size = min(len(inputs), N)
		loss = network.train_on_batch(
				np.array([inputs]),
				np.array([targets]))
		print(loss)

		f = time.time()
		t += 1
		#print("{:04.2f},{:04.2f},{:04.2f},{:04.2f},{:04.2f}".format(b-a,c-b,d-c,e-d,f-e))

	print("Episode finished after {} timesteps".format(t+1))
	epsilon = epsilon*.99 + 0.01**2
	done = False
