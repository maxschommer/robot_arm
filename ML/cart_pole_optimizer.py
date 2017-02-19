#cart_pole_optimizer.py
import numpy as np
import random as rng
import math
import gym
from cart_pole import DQNAgent

param_lims = [(1000,100000),(0.01,0.5),(0.1,1.0),(0.001,0.2),(0.001,0.1),(0.0001,0.01),(16,64)]

env = gym.make('CartPole-v0')

while True:
	params = []
	for minp, maxp in param_lims:
		minl, maxl = math.log(minp), math.log(maxp)
		param = math.exp(rng.random()*(maxl-minl)+minl)
		params.append(param)

	scores=[]
	for i in range(5):
		agent = DQNAgent(env, mem_size=int(params[0]), gamma=1-params[1], epsilon=params[2], epsilon_decay=1-params[3],
			epsilon_min=params[4], learning_rate=params[5], batch_size=int(params[6]))
		score = agent.train(verbose=False)
		scores.append(score)

	print("{:07.3f} | {}".format(np.mean(scores), agent))
