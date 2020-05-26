# -*- coding: utf-8 -*-
# -------------------------------
# Author: Zikang Xiong
# Email: zikangxiong@gmail.com
# Date:   2019-11-01 15:54:21
# Last Modified by:   Zikang Xiong
# Last Modified time: 2019-11-01 16:09:31
# -------------------------------
from plot import plot3d_anim

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import TVTDRL
ROOT = os.path.dirname(os.path.abspath(TVTDRL.__file__))

from TVTDRL.metrics.simulation import simulation_with_nn

import numpy as np

from stable_baselines.ddpg.policies import FeedForwardPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.ddpg.noise import OrnsteinUhlenbeckActionNoise
from stable_baselines import DDPG

import gym
import CoRec

def get_model(load=True):
    # Create and wrap the environment
    env = gym.make('F16GCAS-v0')
    env = DummyVecEnv([lambda: env])

    # Custom MLP policy of two layers of size 16 each
    class CustomPolicy(FeedForwardPolicy):
        def __init__(self, *args, **kwargs):
            super(CustomPolicy, self).__init__(*args, **kwargs,
                                               layers=[128, 128],
                                               layer_norm=False,
                                               feature_extraction="mlp")

    model = DDPG(CustomPolicy, env, verbose=1)
    # # Train the agent
    if load == False:
        model.learn(total_timesteps=1000000)
        model.save(ROOT+"/trained_models/TDRL/f16/128_128")
    model = DDPG.load(ROOT+"/trained_models/TDRL/f16/ddpg/128_128.pkl", policy=CustomPolicy)

    return model

def plot_nn():
    nn_model = get_model()
    env = gym.make("F16GCAS-v0")
    # x0 = np.array([env.env.Vt_low, env.env.alpha_low, env.env.beta_low, env.env.phi_low, env.env.theta_low, env.env.psi_low, 0, 0, 0, 0, 0, env.env.alt_low, env.env.power_low, 0, 0, 0])
    x0 =  np.array(
          [5.4404559e+02,  3.5067257e-02,  0.0000000e+00,  7.2620559e-01,
          -1.2658756e+00, -7.3132914e-01,  0.0000000e+00,  0.0000000e+00,
           0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  3.4306707e+03,
           8.9695072e+00,  0.0000000e+00,  0.0000000e+00,  0.0000000e+00])

    trace, reward = simulation_with_nn(env, 2000, nn_model, times=1, x0=x0)
    plot3d_anim(env.env.times+env.env.times[-1:]*(2000-len(env.env.times)), \
        env.env.states+env.env.states[-1:]*(2000-len(env.env.states)),\
        env.env.modes+env.env.modes[-1:]*(2000-len(env.env.modes)), \
        env.env.ps_list+env.env.ps_list[-1:]*(2000-len(env.env.ps_list)), \
        env.env.Nz_list+env.env.Nz_list[-1:]*(2000-len(env.env.Nz_list)), filename="unsafe.gif")
    trace, reward = simulation_with_nn(env, 2000, nn_model, times=1)
    assert len(env.env.states[:2000]) == 2000
    plot3d_anim(env.env.times, env.env.states, env.env.modes, \
        env.env.ps_list, env.env.Nz_list, filename="safe.gif")


if __name__ == "__main__":
    plot_nn()
