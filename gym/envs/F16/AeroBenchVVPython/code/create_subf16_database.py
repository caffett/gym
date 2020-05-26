# -*- coding: utf-8 -*-
# -------------------------------
# Author: Zikang Xiong
# Email: zikangxiong@gmail.com
# Date:   2019-10-02 16:59:51
# Last Modified by:   Zikang Xiong
# Last Modified time: 2019-10-03 09:39:35
# -------------------------------
from subf16_model import subf16_model
from Morellif16 import Morellif16
import numpy as np

# x_shape = (13, )
# u_shape = (4, )
input_shape = (13, )
output_shape = (6, )

def random_create(dataset_size=100000):
    # input_xs = 10*np.random.random((dataset_size, )+x_shape)
    # input_us = 10*np.random.random((dataset_size, )+u_shape)
    input_xs = 10*np.random.random((dataset_size, )+input_shape)

    # xcg_mult = 1.0 # center of gravity multiplier
    # val = 1.0      # other aerodynmic coefficient multipliers
    # cxt_mult = val
    # cyt_mult = val
    # czt_mult = val
    # clt_mult = val
    # cmt_mult = val
    # cnt_mult = val
    
    # multipliers = (xcg_mult, cxt_mult, cyt_mult, czt_mult, clt_mult, cmt_mult, cnt_mult)

    # low effeciency
    output_xs = []
    for i in range(dataset_size):
        # output_x, Nz, Ny, _, _ = morelli_model(input_xs[i], input_us[i], model="morelli", multipliers=multipliers)
        # output_xs.append(np.concatenate([output_x, [Ny]], axis=0))
        output = Morellif16(*input_xs[i])
        output_xs.append(output)
    output_xs = np.array(output_xs)

    np.save("morelli_input_x", input_xs)
    # np.save("morelli_input_us", input_us)
    np.save("morelli_output_x", output_xs)

def load(path="./"):
#     input_xs = np.load(path+"morelli_input_xs.npy")
#     input_us = np.load(path+"morelli_input_us.npy")
#     output_xs = np.load(path+"morelli_output_xs.npy")
    input_xs = np.load("./morelli_input_x.npy")
    output_xs = np.load("./morelli_output_x.npy")

    return input_xs, output_xs
#     return input_xs, input_us, output_xs

if __name__ == "__main__":
    random_create(10000)
    # print(load())

