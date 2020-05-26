# -*- coding: utf-8 -*-
# -------------------------------
# Author: Zikang Xiong
# Email: zikangxiong@gmail.com
# Date:   2019-06-28 10:01:26
# Last Modified by:   Zikang Xiong
# Last Modified time: 2019-07-01 11:49:29
# -------------------------------
import gym
import CoRec

import numpy as np

from stable_baselines.common.base_class import BaseRLModel
from stable_baselines.gail import generate_expert_traj

import tensorflow as tf

class LQRModel(BaseRLModel):
    def __init__(self):
        # Longitudinal Gains
        K_lqr_long = np.array([[-156.8801506723475, -31.037008068526642, -38.72983346216317]], dtype=float)

        # Lateral Gains
        K_lqr_lat = np.array([[30.511411060051355, -5.705403676148551, -9.310178739319714, \
                                                    -33.97951344944365, -10.652777306717681], \
                              [-22.65901530645282, 1.3193739204719577, -14.2051751789712, \
                                                    6.7374079391328845, -53.726328142239225]], dtype=float)

        self.K_lqr = np.zeros((3, 8))
        self.K_lqr[:1, :3] = K_lqr_long
        self.K_lqr[1:, 3:] = K_lqr_lat

    def get_env(self):
        env = gym.make("F16GCAS-v0")
        return env

    def predict(self, obs, *args, **kwargs):
        return np.dot(-self.K_lqr, obs), None

    def _get_pretrain_placeholders(self):
        return None
    def action_probability(self):
        return None
    def get_parameter_list(self):
        return None
    def learn(self):
        return None
    def load(self):
        return None
    def save(self):
        return None
    def setup_model(self):
        return None

def generate():
    model = LQRModel()
    generate_expert_traj(model, save_path="./lqr_export.npz", env=None, n_timesteps=0, n_episodes=10)

##################
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import gym, CoRec
import numpy as np

import TVTDRL
ROOT = os.path.dirname(os.path.abspath(TVTDRL.__file__))

from stable_baselines.gail import ExpertDataset

def train_agent_with_a2c(load=False):
    from stable_baselines.common.policies import MlpPolicy
    from stable_baselines.common.vec_env import SubprocVecEnv
    from stable_baselines import A2C

    # multiprocess environment
    n_cpu = 4
    env = SubprocVecEnv([lambda: gym.make('F16GCAS-v0') for i in range(n_cpu)])
    env = gym.make("F16GCAS-v0")

    class CustomPolicy(MlpPolicy):
        def __init__(self, *args, **kwargs):
            super(CustomPolicy, self).__init__(*args, **kwargs,
                                               layers=[128, 128])
    if not load:
        model = A2C(env=env, verbose=1, policy=CustomPolicy)
        # model.learn(total_timesteps=1000000)
        ExpData = ExpertDataset("./lqr_export.npz")
        model.pretrain(ExpData, n_epochs=100)
    else:
        model = A2C.load(ROOT+"/trained_models/TDRL/f16/a2c/128_128", env=env)
        with model.graph.as_default():
            for i in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='model/pi/'):
                print(i)

    return model


def trian_agent_with_gail(load):
    from stable_baselines.common.policies import MlpPolicy
    from stable_baselines import GAIL

    env = gym.make("F16GCAS-v0")

    class CustomPolicy(MlpPolicy):
        def __init__(self, *args, **kwargs):
            super(CustomPolicy, self).__init__(*args, **kwargs,
                                               layers=[128, 128])
    if not load:
        ExpData = ExpertDataset("./lqr_export.npz")
        model = GAIL(CustomPolicy, env, ExpData, verbose=1)
        model.learn(total_timesteps=1000000)
        model.save(ROOT+"/trained_models/TDRL/f16/gail/128_128")
    else:
        # with model.graph.as_default():
        #     for i in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='model/pi/'):
        #         print(i)
        model = GAIL.load(ROOT+"/trained_models/TDRL/f16/gail/128_128", env=env)
        with model.graph.as_default():
            print(tf.all_variables())

    return model

def train_agent_with_ddpg(load):
    from stable_baselines.ddpg.policies import FeedForwardPolicy
    from stable_baselines.common.vec_env import DummyVecEnv
    from stable_baselines.ddpg.noise import OrnsteinUhlenbeckActionNoise
    from stable_baselines import DDPG

    # Create and wrap the environment
    env = gym.make('F16GCAS-v0')
    env = DummyVecEnv([lambda: env])

    # the noise objects for DDPG
    n_actions = env.action_space.shape[-1]
    param_noise = None
    action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=float(0.01) * np.ones(n_actions))

    # Custom MLP policy of two layers of size 16 each
    class CustomPolicy(FeedForwardPolicy):
        def __init__(self, *args, **kwargs):
            super(CustomPolicy, self).__init__(*args, **kwargs,
                                               layers=[128, 128],
                                               layer_norm=False,
                                               feature_extraction="mlp")

    model = DDPG(CustomPolicy, env, verbose=1, action_noise=action_noise)

    if not load:
        ExpData = ExpertDataset("./lqr_export.npz")
        model.pretrain(ExpData, n_epochs=100)
        model.save(ROOT+"/trained_models/TDRL/f16/ddpg/128_128")
    else:
        model = DDPG.load(ROOT+"/trained_models/TDRL/f16/ddpg/128_128", policy=CustomPolicy, env=env)

    return model

def test_model(model):
    env = gym.make('F16GCAS-v0')
    obs = env.reset()

    count = 0
    # pre_mode = "Unknow"
    action_list1 = []
    action_list2 = []
    state_list1 = []
    state_list2 = []
    while True:
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        action_list1.append(action)
        state_list1.append(obs)
        # if env.env.modes[-1] != pre_mode:
        #     print(env.env.modes[-2], len(env.env.modes), env.env.times[-2])
        #     print(env.env.states[-2])
        #     print(env.env.modes[-1], len(env.env.modes)-1, env.env.times[-1])
        #     print(env.env.states[-1])
        #     print("=======")
        #     pre_mode = env.env.modes[-1]
        count += 1
        if dones:
            print(env.env.states[-1])
            break

    obs = env.reset()
    while True:
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        action_list2.append(action)
        state_list2.append(obs)
        # if env.env.modes[-1] != pre_mode:
        #     print(env.env.modes[-2], len(env.env.modes), env.env.times[-2])
        #     print(env.env.states[-2])
        #     print(env.env.modes[-1], len(env.env.modes)-1, env.env.times[-1])
        #     print(env.env.states[-1])
        #     print("=======")
        #     pre_mode = env.env.modes[-1]
        count += 1
        if dones:
            print(env.env.states[-1])
            break

    print(np.sum(np.array(action_list1)-np.array(action_list2)))
    print(np.sum(np.array(state_list1)==np.array(state_list2)))

    from TVTDRL.metrics.draw import draw_state_list
    for i in range(16):
        draw_state_list(env.env.states, state_index=[i])

    # animFilename = 'gcas.gif'
    # from CoRec.envs.F16.AeroBenchVVPython.code.plot import plot3d_anim
    # if animFilename is not None:
    #     plot3d_anim(env.env.times, env.env.states, env.env.modes, env.env.ps_list, env.env.Nz_list, filename=animFilename)

##########################

if __name__ == "__main__":
    # generate()
    model = train_agent_with_ddpg(load=True)
    test_model(model)
