# -*- coding: utf-8 -*-
# -------------------------------
# Author: Zikang Xiong
# Email: zikangxiong@gmail.com
# Date:   2019-09-27 14:54:22
# Last Modified by:   Zikang Xiong
# Last Modified time: 2019-10-13 23:38:43
# -------------------------------
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import losses
from tensorflow.keras.models import load_model

import gym
import CoRec

import TVTDRL.algorithms.BaselinesModel.baselines_model as bm
# import TVTDRL.algorithms.VRL.vrl_neural as vn

def create_dataset_f16(env, controller, step, iteration, prefix="", postfix=""):
	input_x = []
	input_u = []
	output_x = []

	for it in range(iteration):
		env.reset()
		if it % 100 == 0:
			print("{}/{}".format(it, iteration))
		for s in range(step):
			input_x.append(env.env.states[-1])
			u_1_3, _ = controller.predict(env.env.x_ctrl)
			env.step(u_1_3)
			input_u.append(env.env.ut)
			output_x.append(env.env.states[-1])

	np.save("./nn_model/data/"+prefix+"input_x"+postfix+".npy", np.array(input_x))
	np.save("./nn_model/data/"+prefix+"input_u"+postfix+".npy", np.array(input_u))
	np.save("./nn_model/data/"+prefix+"output_x"+postfix+".npy", np.array(output_x))


def create_dataset_vrl(env, controller, step, iteration):
	input_x = []
	input_u = []
	output_xd = []

	for it in range(iteration):
		xt = env.reset()
		if it % 100 == 0:
			print("{}/{}".format(it, iteration))
		for s in range(step):
			input_x.append(xt)
			ut, _ = controller.predict(xt)
			xt, _, _, _ = env.step(ut)
			input_u.append(ut)
			output_xd.append(xt)

	np.save("vrl_input_x.npy", np.array(input_x))
	np.save("vrl_input_u.npy", np.array(input_u))
	np.save("vrl_output_xd.npy", np.array(output_xd))


def read_dataset(path, prefix="", postfix="", shuffle=True):
	input_x = np.load(path+"/"+prefix+"input_x"+postfix+".npy")
	input_u = np.load(path+"/"+prefix+"input_u"+postfix+".npy")
	output_x = np.load(path+"/"+prefix+"output_x"+postfix+".npy")
	concat_array = np.concatenate([input_x, input_u, output_x], axis=-1)
	if shuffle:
		np.random.shuffle(concat_array)

	input_xu = concat_array[:, :-output_x.shape[-1]]
	output_xd = concat_array[:, -output_x.shape[-1]:]

	return input_xu, output_xd

def create_fully_connected_approximator(input_size, hidden_structure, output_size):
	with tf.variable_scope("environment_approximator", reuse=True):
		approx = tf.keras.Sequential()
		approx.add(layers.BatchNormalization(input_shape=(input_size,)))
		for neuron_num in hidden_structure:
			approx.add(layers.Dense(neuron_num, activation='elu'))
		approx.add(layers.Dense(output_size, activation='linear', name="xt_plus_1"))
		
		approx.compile(optimizer=keras.optimizers.Adam(0.001), 
						loss='mean_squared_error', #losses.mean_square_error, 
						metrics=['mae'])

	return approx


def train_approximator(structure, model_path=None):
	if model_path is not None:
		approx = load_model(model_path)
	else:
		input_size = structure[0]
		hidden_structure = structure[1] 
		output_size = structure[2]
		approx = create_fully_connected_approximator(input_size, hidden_structure, output_size)
	
	input_set, output_set = read_dataset(".", prefix="morelli_")#read_dataset("./nn_model/data", postfix="_subf16")
	approx.fit(input_set, output_set, epochs=1000, batch_size=128)

	if model_path is None:
		approx.save("approx_morelli.model")
	else:
		approx.save(model_path)


## This part is for 3000 step approximation
def train_2000_steps_apporximator(structure, model_path=None):
	if model_path is not None:
		approx = load_model(model_path)
	else:
		input_size = structure[0]
		hidden_structure = structure[1] 
		output_size = structure[2]
		approx = create_fully_connected_approximator(input_size, hidden_structure, output_size)
	
	input_set, output_set = read_dataset("./nn_model/data", prefix="f16_", postfix="_2000_1000", shuffle=False)
	input_set = input_set.reshape((1000, 2000, 20))
	output_set = output_set.reshape((1000, 2000, 16))
	# print(input_set.shape)
	# print(output_set.shape)
	training_input_set = input_set[:, :10, :].reshape((10000, 20))[:, :16]
	training_output_set = output_set[:, -10:, :].reshape((10000, 16))
	# print(training_input_set.shape)
	# print(training_output_set.shape)
	# assert False
	print("input:\n", training_input_set[:, :])
	print("expect output: \n", training_output_set[:, :])
	approx.fit(training_input_set[:, :], training_output_set[:, :], epochs=500, batch_size=128)

	if model_path is None:
		approx.save("nn_model/trained_model/approx.model")
	else:
		approx.save(model_path)

def inspect_trained_model():
	input_xu, output_xd = read_dataset("./nn_model/data", prefix="f16_", postfix="_2000_1000", shuffle=False)
	approx = load_model("nn_model/trained_model/approx.model")
	input_list_index = [i*2000 for i in range(1)]
	output_list_index = [(i+1)*2000 for i in range(1)]

	predict_res = approx.predict(input_xu[input_list_index])
	print("-------")
	for p, o in zip(predict_res, output_xd[output_list_index]):
		print("original: \n", o)
		print("predict: \n", p)
		print("diff: \n", p-o)
		print("-------")


if __name__ == "__main__":
	model_name = "F16GCAS-v0"
	nn_structure = [16, [256, 256, 256], 16]
	# model_name = "VrlHelicopter-v0"
	# env = gym.make(model_name)
	# controller = bm.get_by_name(model_name)
	# create_dataset_f16(env, controller, 2000, 5000, prefix="f16_", postfix="_2000_5000")
	# create_dataset_vrl(env, controller, 2000, 10)
	# read_dataset(".", prefix="vrl_")
	# train_approximator()
	train_2000_steps_apporximator(nn_structure)
	inspect_trained_model()
	# naive_test()

