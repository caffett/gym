# -*- coding: utf-8 -*-
# -------------------------------
# Author: Zikang Xiong
# Email: zikangxiong@gmail.com
# Date:   2019-10-30 10:08:05
# Last Modified by:   Zikang Xiong
# Last Modified time: 2019-11-20 11:18:18
# -------------------------------
import argparse
import gym
import os
ROOT = os.path.dirname(os.path.abspath(gym.__file__))+"/envs/env_approx/"

import numpy as np
import math
import itertools

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.losses import mean_absolute_percentage_error
from tensorflow.keras.layers import BatchNormalization, RepeatVector, \
									LSTM, Dense, TimeDistributed, GaussianNoise, Input

from recurrentshop import RecurrentModel, RecurrentSequential
from recurrentshop.cells import LSTMCell

from scout.Utils.NNTools import zoo_model, old_ddpg_model

from scout.Utils.StateCompress.compressor import Compressor 

def create_fully_connected_approximator(input_size, hidden_structure, output_size):
	"""
		As the function name
	"""
	with tf.variable_scope("environment_approximator", reuse=True):
		approx = keras.Sequential()
		approx.add(keras.layers.BatchNormalization(input_shape=(input_size,)))
		for neuron_num in hidden_structure:
			approx.add(keras.layers.Dense(neuron_num, activation='relu'))
		approx.add(keras.layers.Dense(output_size, activation='linear', name="xt_plus_1"))
		
		approx.compile(optimizer=keras.optimizers.Adam(0.001), 
						loss=mean_absolute_percentage_error, 
						metrics=['mae', mean_absolute_percentage_error])

	return approx

def create_rnn_approximator(state_size, chain_length, \
							noise_stddev=0.0, load_path=None, learning_rate=1e-3):
	magic_number = (int(math.sqrt(state_size))+1)*32

	if load_path is not None:
		approx = load_model(load_path)
	else:
		with tf.variable_scope("rnn_environment_approximator", reuse=True):

			# approx = RecurrentSequential(readout="readout_only", teacher_force=True, return_sequences=True)
			# approx.add(RepeatVector(chain_length, input_shape=(state_size,)))
			# approx.add(LSTMCell(magic_number, input_dim=(chain_length, state_size)))
			# # approx.add(TimeDistributed(Dense(state_size, activation="linear")))

			# x = Input((state_size, ))
			# y_true = Input((chain_length, state_size))
			# y = approx(x, ground_truth=y_true)
			# training_model = Model([x, y_true], y)

			approx = keras.Sequential()
			# approx.add(BatchNormalization(input_shape=(None, state_size)))
			# approx.add(GaussianNoise(noise_stddev, input_shape=(None, state_size)))
			approx.add(RepeatVector(chain_length))
			approx.add(LSTM(magic_number, return_sequences=True))
			approx.add(LSTM(magic_number, return_sequences=True))
			approx.add(TimeDistributed(Dense(magic_number, activation="relu")))
			approx.add(keras.layers.Dense(magic_number, activation="relu"))
			approx.add(keras.layers.Dense(magic_number, activation="relu"))
			approx.add(keras.layers.Dense(magic_number, activation="relu"))
			approx.add(TimeDistributed(Dense(state_size, activation="linear")))

	approx.compile(optimizer=keras.optimizers.Adam(learning_rate), 
					loss="mae", #mean_absolute_percentage_error, 
					metrics=['mae'])

	return approx


# All dataset generated with deterministic model
def nn_generate_reward_dataset(model_name, algo=None, iteration=1e3, approx_step=150, max_step=200, compress=False, nn_model=None):
	if not os.path.exists(ROOT+model_name):
		os.makedirs(ROOT+model_name)

	if nn_model is None:
		assert not (algo is None)
		nn_model = zoo_model.load_model(model_name, algo)

	env = gym.make(model_name)

	if compress == True:
		cp = Compressor(env.env.initial_state.low, env.env.initial_state.high)
	
	input_x = []
	output_reward = []

	for i in range(iteration):
		obs = env.reset()
		rewards = []
		cumulative_rewards = []
		states = [np.array(env.env.state)[cp.nonzero_index] if compress else env.env.state]

		for i in range(max_step):
			u, _ = nn_model.predict(obs, deterministic=True)
			obs, reward, done, _ = env.step(u)
			if i < max_step - approx_step - 1:
				states.append(np.array(env.env.state)[cp.nonzero_index] if compress else env.env.state)
			rewards.append(reward)

		input_x.append(states)

		for i in range(max_step-approx_step+1):
			cumulative_rewards.append(np.sum(rewards[i: approx_step+i]))

		output_reward.append(cumulative_rewards)

	input_x = np.array(input_x)
	output_reward = np.array(output_reward)

	print("------------ "+model_name+" dataset info ------------")
	print("input x shape: ", input_x.shape)
	print("output reward shape: ", output_reward.shape)
	print("--------------------------------------------------")

	np.save(ROOT+model_name+"/"+algo+"_ra_input_x_"+"%.0e"%iteration, input_x)
	np.save(ROOT+model_name+"/"+algo+"_ra_output_reward_"+"%.0e"%iteration, output_reward)

def nn_trace_generator(model_name, algo=None, iteration=1e3, max_step=200, nn_model=None):
	if not os.path.exists(ROOT+model_name):
		os.makedirs(ROOT+model_name)

	if nn_model is None:
		if model_name[:3] == "Vrl" or model_name[:3] == "F16": # old model
			nn_model = old_ddpg_model.load(model_name)
		else: # zoo model
			assert not (algo is None)
			nn_model = zoo_model.load_model(model_name, algo)

	env = gym.make(model_name)

	traces = []

	for i in range(iteration):
		obs = env.reset()
		states = [np.array(env.env.state)]

		for i in range(max_step):
			u, _ = nn_model.predict(obs, deterministic=True)
			obs, reward, done, _ = env.step(u)
			states.append(env.env.state)
			if done:
				break

		traces.append(states)

	np.save(ROOT+model_name+"/"+algo+"_trace_"+"%.0e"%iteration, np.array(traces))

def nn_generate_vertex_reward_dataset(model_name, algo, approx_step, max_step, nn_model=None):
	# only support compress mode now
	assert approx_step == max_step # TODO: Refactor this

	if not os.path.exists(ROOT+model_name):
		os.makedirs(ROOT+model_name)

	if nn_model is None:
		assert not (algo is None)
		nn_model = zoo_model.load_model(model_name, algo)

	env = gym.make(model_name)
	cp = Compressor(env.env.initial_space.low, env.env.initial_space.high)

	input_x = []
	output_reward = []

	low_high_pairs = []
	compressed_state = cp.compress()
	for i in range(np.sum(cp.nonzero_index)):
		low_high_pairs.append([compressed_state[0][i], compressed_state[1][i]])

	if len(low_high_pairs) > 1:
		vertices = [i for i in itertools.product(*low_high_pairs)]
	elif len(low_high_pairs) == 1:
		vertices = [np.array(i) for i in low_high_pairs[0]]

	vertices.append((compressed_state[0]+compressed_state[1])/2)
	
	for compressed_x0 in vertices:
		obs = env.reset(x0=cp.decompress_any(compressed_x0))
		rewards = []
		cumulative_rewards = []
		states = [np.array(env.env.state)[cp.nonzero_index]]

		for i in range(max_step):
			u, _ = nn_model.predict(obs, deterministic=True)
			obs, reward, done, _ = env.step(u)
			if i < max_step - approx_step - 1:
				states.append(np.array(env.env.state)[cp.nonzero_index])
			rewards.append(reward)

		input_x.append(states)

		for i in range(max_step-approx_step+1):
			cumulative_rewards.append(np.sum(rewards[i: approx_step+i]))

		print(compressed_x0)
		print(cumulative_rewards[-1])
		output_reward.append(cumulative_rewards)

	input_x = np.array(input_x)
	output_reward = np.array(output_reward)

	print("------------ "+model_name+" dataset info ------------")
	print("input x shape: ", input_x.shape)
	print("output reward shape: ", output_reward.shape)
	print("--------------------------------------------------")

	np.save(ROOT+model_name+"/"+algo+"_vra_input_x", input_x)
	np.save(ROOT+model_name+"/"+algo+"_vra_output_reward", output_reward)


def generate_reward_approximator(model_name, prefix="", postfix="", approx_step=None):
	input_x = np.load(ROOT+model_name+"/"+prefix+"input_x"+postfix+".npy")
	output_reward = np.load(ROOT+model_name+"/"+prefix+"output_reward"+postfix+".npy")

	input_size = input_x.shape[-1]
	output_size = 1
	# increase the neural network's size according to the size of input
	# magic_number = int(math.sqrt(input_size))+1
	# hidden_structure = []
	# for i in range(magic_number):
	# 	hidden_structure.append(32*magic_number)
	hidden_structure = [128, 128, 64, 64, 64]

	input_set = input_x.reshape([-1, input_size])
	output_set = output_reward.reshape([-1, output_size])

	postfix += "_approx"+str(approx_step)

	approx = create_fully_connected_approximator(input_size, hidden_structure, output_size)

	batch_size = int(np.clip(len(input_set)/1000, 1, 256))
	approx.fit(input_set, output_set, epochs=500, batch_size=batch_size)

	approx.save(ROOT+model_name+"/"+prefix+"approx"+postfix+".model")


def generate_rnn_approximator(model_name, prefix="", postfix="", 
	max_step=200, approx_step=1, load_path=None, learning_rate=1e-4):
	traces = np.load(ROOT+model_name+"/"+prefix+"_trace_"+postfix+".npy")
	state_size = traces[-1].shape[-1]
	approx = create_rnn_approximator(state_size=state_size, chain_length=max_step, 
									load_path=load_path, learning_rate=learning_rate)

	input_set = []
	output_set = []
	trace_index = np.array([i*approx_step for i in range(int(max_step/approx_step))])
	for i in range(len(traces)):
		input_set.append(traces[i][trace_index[0]])
		output_set.append(traces[i][trace_index])

	# approx.summary()
	input_set = np.array(input_set)#[:1]
	output_set = np.array(output_set)#[:1]

	batch_size = int(np.clip(len(input_set)/1000, 1, 128))
	approx.fit(input_set, output_set, epochs=10, batch_size=batch_size)
	approx.save(ROOT+model_name+"/"+prefix+"_rnn_approx_"+postfix+".model")


def reward_approx_generator(model_name, algo, iteration=None, approx_step=None, max_step=None, compress=False, vertices=False):
	if vertices:
		nn_generate_vertex_reward_dataset(model_name, algo, approx_step, max_step)
		generate_reward_approximator(model_name, prefix=algo+"_vra_", approx_step=approx_step)
	else:
		assert iteration is not None
		nn_generate_reward_dataset(model_name, algo, iteration=int(iteration), approx_step=approx_step, max_step=max_step)
		generate_reward_approximator(model_name, prefix=algo+"_ra_", postfix="_%.0e"%iteration, approx_step=approx_step)

def rnn_approximator_generator(model_name, algo, iteration, max_step, approx_step):
	# nn_trace_generator(model_name, algo, iteration=iteration, max_step=max_step)
	generate_rnn_approximator(model_name, prefix=algo, postfix="%.0e"%iteration, max_step=max_step, approx_step=approx_step)

if __name__ == "__main__":
	generate_rnn_approximator("Pendulum-v0", prefix="a2c", max_step=200, postfix="%.0e"%50, approx_step=1, 
		# load_path="/home/zxiong/development/docker_share/scout/envs/gym/gym/envs/env_approx/Pendulum-v0/a2c_rnn_approx_5e+01.model", 
		learning_rate=1e-4)
	# help_text = \
	# """
	# Script used for building approximator
	# """
	# parser = argparse.ArgumentParser(description=help_text)
	# parser.add_argument("--env", "-e", help="environment name", type=str)
	# parser.add_argument("--algo", "-A", help="reinforcement learning algorithm", type=str)
	# parser.add_argument("--iter", "-i", help="compress initial state or not", type=int, default=1000)
	# parser.add_argument("--max_step", "-m", help="max step of environment", type=int)
	# parser.add_argument("--approx_step", "-a", help="approximate step", type=int)
	# parser.add_argument("--compress", "-c", help="compress initial state or not", type=bool, default=False)
	# parser.add_argument("--vertices", "-v", help="sample initial state in vetices or not", type=bool, default=False)

	# args = parser.parse_args()

	# # reward_approx_generator(args.env, args.algo, args.iter, args.approx_step,\
	# # 						args.max_step, args.compress, args.vertices)
	# rnn_approximator_generator(args.env, args.algo, args.iter, args.max_step, args.approx_step)
	

# RA_args = [
# 	# ["Pendulum-v0", "a2c", 1e3, 200-50, 200], 
# 	# ["Pendulum-v0", "acktr", 1e3, 200-50, 200],
# 	# ["Pendulum-v0", "ppo2", 1e3, 200-50, 200],
# 	# ["Pendulum-v0", "trpo", 1e3, 200-50, 200],
# 	# ["Pendulum-v0", "ddpg", 1e3, 200-50, 200], 
# 	# ["Pendulum-v0", "sac", 1e3, 200-50, 200],
# 	# ["Pendulum-v0", "td3", 1e3, 200-50, 200],

# 	# ["CartPole-v1", "a2c", 1e3, 500-100, 500], 
# 	# ["CartPole-v1", "acktr", 1e3, 500-100, 500], 
# 	# ["CartPole-v1", "ppo2", 1e3, 500-100, 500],
# 	# ["CartPole-v1", "trpo", 1e3, 500-100, 500], 
# 	# ["CartPole-v1", "acer", 1e3, 500-100, 500], 
# 	# ["CartPole-v1", "dqn", 1e3, 500-100, 500], 

# 	["MountainCar-v0", "a2c", 1e3, 200-50, 200], 
# 	["MountainCar-v0", "acktr", 1e3, 200-50, 200], 
# 	["MountainCar-v0", "ppo2", 1e3, 200-50, 200], 
# 	["MountainCar-v0", "trpo", 1e3, 200-50, 200],
# 	["MountainCar-v0", "acer", 1e3, 200-50, 200], 
# 	# ["MountainCar-v0", "dqn", 1e3, 200-50, 200],

# 	["Acrobot-v1", "a2c", 1e3, 500-100, 500], 
# 	["Acrobot-v1", "acktr", 1e3, 500-100, 500],
# 	["Acrobot-v1", "ppo2", 1e3, 500-100, 500],  
# 	# ["Acrobot-v1", "trpo", 1e6, 500, 500], 
# 	["Acrobot-v1", "acer", 1e3, 500-100, 500],
# 	# ["Acrobot-v1", "dqn", 1e3, 500-100, 500],

# 	["MountainCarContinuous-v0", "a2c", 1e3, 999-200, 999], 
# 	["MountainCarContinuous-v0", "acktr", 1e3, 999-200, 999], 
# 	["MountainCarContinuous-v0", "ppo2", 1e3, 999-200, 999], 	
# 	["MountainCarContinuous-v0", "trpo", 1e3, 999-200, 999], 
# 	["MountainCarContinuous-v0", "ddpg", 1e3, 999-200, 999], 
# 	["MountainCarContinuous-v0", "sac", 1e3, 999-200, 999], 	
# 	["MountainCarContinuous-v0", "td3", 1e3, 999-200, 999], 
# ]


# if __name__ == "__main__": 
# 	for args in RA_args:
# 		reward_approx_template(*args)
