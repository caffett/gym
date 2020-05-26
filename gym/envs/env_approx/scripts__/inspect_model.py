# -*- coding: utf-8 -*-
# -------------------------------
# Author: Zikang Xiong
# Email: zikangxiong@gmail.com
# Date:   2019-10-25 09:42:34
# Last Modified by:   Zikang Xiong
# Last Modified time: 2019-11-14 12:18:01
# -------------------------------
import numpy as np
from scout.Utils.Metrics.draw import draw_3d

import gym
from tensorflow.keras.models import load_model

def inspect_pendulum_reward():
	X = []
	Y = []
	Z = []

	x = np.load("Pendulum-v0/a2c_ra_input_x_1e+03.npy").reshape([-1, 2])
	r = np.load("Pendulum-v0/a2c_ra_output_reward_1e+03.npy").reshape([-1, 1])

	for i in range(len(x[:5000])):
		X.append(x[i][0])
		Y.append(x[i][1])
		Z.append(r[i])

	draw_3d(X, Y, Z)

def inspect_pendulum_reward_approx():
	x = np.load("Pendulum-v0/a2c_ra_input_x_1e+03.npy").reshape([-1, 2])
	nn_model = load_model("Pendulum-v0/a2c_ra_approx_1e+03_approx150.model")
	r  = nn_model.predict(x)
	X = []
	Y = []
	Z = []

	for i in range(len(x[:5000])):
		X.append(x[i][0])
		Y.append(x[i][1])
		Z.append(r[i])

	draw_3d(X, Y, Z)

from scout.Utils.Metrics.draw import draw_state_list

def inspect_rnn_approx():
	traces = np.load("/home/zxiong/development/docker_share/scout/envs/gym/gym/envs/env_approx/Pendulum-v0/a2c_trace_5e+01.npy")
	
	input_set = []
	output_set = []
	for i in range(len(traces)):
		input_set.append(traces[i][:-1])
		output_set.append(traces[i][1:])

	nn_model = load_model("/home/zxiong/development/docker_share/scout/envs/gym/gym/envs/env_approx/Pendulum-v0/a2c_rnn_approx_5e+01.model")
	nn_model.summary()
	print(np.array(input_set[0]).shape)

	pred_res = nn_model.predict(np.array(input_set))
	# print(pred_res[0])
	for res, out in zip(pred_res, output_set):
		draw_state_list(res)
		draw_state_list(out)

def inspect_rnn_approx_step_by_step():
	traces = np.load("/home/zxiong/development/docker_share/scout/envs/gym/gym/envs/env_approx/Pendulum-v0/a2c_trace_5e+01.npy")
	
	input_set = []
	output_set = []
	for i in range(len(traces)):
		input_set.append(traces[i][:-1])
		output_set.append(traces[i][1:])

	nn_model = load_model("/home/zxiong/development/docker_share/scout/envs/gym/gym/envs/env_approx/Pendulum-v0/a2c_rnn_approx_5e+01.model")
	nn_model.summary()

	pred_res = []
	for i in range(5):#len(traces)):
		input_seq = [np.array(input_set[i][:1])]
		for j in range(200):
			output = nn_model.predict(np.array([input_seq[-1]]))
			input_seq.append(output[-1])
		print(np.array(input_seq).shape)
		seq = np.array(input_seq)
		seq = seq.reshape(201, 2)
		pred_res.append(seq.tolist())

	for res, out in zip(pred_res, traces[:5]):
		print(res[:20])
		print(out[:20])
		draw_state_list(res)
		draw_state_list(out)


if __name__ == "__main__":
	# inspect_pendulum_reward()
	# inspect_pendulum_reward_approx()
	# inspect_rnn_approx()
	inspect_rnn_approx_step_by_step()