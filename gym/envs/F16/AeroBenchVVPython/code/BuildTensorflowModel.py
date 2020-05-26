# -*- coding: utf-8 -*-
# -------------------------------
# Author: Zikang Xiong
# Email: zikangxiong@gmail.com
# Date:   2019-06-26 20:24:39
# Last Modified by:   Zikang Xiong
# Last Modified time: 2019-09-17 16:19:10
# -------------------------------
import os
from math import pi as math_pi
import numpy as np
pi = np.float32(math_pi)

import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import gym
import CoRec

import TVTDRL
ROOT = os.path.dirname(os.path.abspath(TVTDRL.__file__))

from stable_baselines import DDPG
from stable_baselines.ddpg.policies import FeedForwardPolicy
from stable_baselines.common.vec_env import DummyVecEnv

from CoRec.envs.F16.AeroBenchVVPython.code.subf16_model import subf16_model_tf

from TVTDRL.metrics.time import timeit

from TVTDRL.metrics.simulation import simulation_with_nn

class F16TFModel:

    def __init__(self, step, opt_learning_rate=0.01):
        self.env = gym.make("F16GCAS-v0")

        self.sim_step = 0.01
        self.step = step
        self.opt_learning_rate = opt_learning_rate

        self.graph = tf.get_default_graph()
        self.sess = tf.Session()

        self.STATE_START = 0
        self.STATE_ROLL = 1
        self.STATE_PULL = 2
        self.STATE_DONE = 3
        with tf.name_scope("high_level_state"):
            self.change_points = [  tf.Variable(0.0, name="start2roll", dtype=tf.float32),
                                    tf.Variable(0.0, name="roll2pull", dtype=tf.float32),
                                    tf.Variable(0.0, name="pull2done", dtype=tf.float32)]
            self.roll_state_placeholder = tf.placeholder(dtype=tf.float32, name="roll_state")
            self.pull_state_placeholder = tf.placeholder(dtype=tf.float32, name="pull_state")
            self.done_state_placeholder = tf.placeholder(dtype=tf.float32, name="done_state")
            self.roll_state_assign = tf.assign(self.change_points[0], self.roll_state_placeholder)
            self.pull_state_assign = tf.assign(self.change_points[1], self.pull_state_placeholder)
            self.done_state_assign = tf.assign(self.change_points[2], self.done_state_placeholder)

        self.nn_model = None
        self._get_weights()

        self.subf16_model_tf = subf16_model_tf

        self.xt_list = []
        self.minimize_tf = None
        self.env.reset()

        self.build_chain()

        init = tf.global_variables_initializer()
        self.sess.run(init)


    # Load the parameters of neural network from pretrained model
    def _get_weights(self):
        class CustomPolicy(FeedForwardPolicy):
            def __init__(self, *args, **kwargs):
                super(CustomPolicy, self).__init__(*args, **kwargs,
                                                   layers=[128, 128],
                                                   layer_norm=False,
                                                   feature_extraction="mlp")
        DVenv = DummyVecEnv([lambda: self.env])
        self.nn_model = DDPG.load(ROOT+"/trained_models/TDRL/f16/ddpg/128_128", policy=CustomPolicy)

        with self.nn_model.graph.as_default():
            # print(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='model/pi'))
            # print(tf.all_variables())
            # train_writer = tf.summary.FileWriter('./neural_network_graph', model.sess.graph)
            wb_list = self.nn_model.sess.run(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='model/pi'))

        self.w_list = []
        self.b_list = []
        count = 0
        with tf.name_scope("neural_controller"):
            for wb in wb_list:
                if count%2 == 0:
                    self.w_list.append(tf.convert_to_tensor(wb, name="w"))
                else:
                    self.b_list.append(tf.convert_to_tensor(wb, name="b"))
                count += 1


    def build_neural_controller(self, x_f16, u_ref_tf):
        with tf.name_scope("neural_controller"):
            x_delta = x_f16 - self.env.env.stable_state
            x_ctrl = tf.gather_nd(x_delta, [[1], [7], [13], [2], [6], [8], [14], [15]])
            u_deg_0_tf = u_ref_tf[3]

            input_x = tf.reshape(x_ctrl, (-1, 8))
            for  w, b in zip(self.w_list[:-1], self.b_list[:-1]):
                input_x = tf.matmul(input_x, w) + b
                input_x = tf.nn.relu(input_x)
            u_deg_123_tf = tf.nn.tanh(tf.matmul(input_x, self.w_list[-1]) + self.b_list[-1])*self.env.action_space.high

            u_deg = tf.concat([[[u_deg_0_tf]], u_deg_123_tf], axis=1)[0]
            u_deg = u_deg + self.env.env.uequil
            u_deg = tf.clip_by_value(u_deg, self.env.env.u_low[:4], self.env.env.u_high[:4])


        return u_deg

    # @timeit
    def update_change_points(self):
        x0 = self.sess.run(self.x0)
        ob = self.env.reset(x0=x0)

        CurrentState = self.STATE_START

        eps_phi = np.deg2rad(5)   # Max roll angle magnitude before pulling g's
        eps_p = np.deg2rad(1)     # Max roll rate magnitude before pulling g's
        path_goal = np.deg2rad(0) # Final desired path angle
        man_start = 2.0

        self.sess.run(self.roll_state_assign,
                      feed_dict={self.roll_state_placeholder: man_start})

        t_count = 1
        done = False
        while not done:
            action, _ = self.nn_model.predict(ob)
            ob, reward, done, _ = self.env.step(action)
            x_f16 = self.env.env.states[-1]

            # Pull out important variables for ease of use
            phi = x_f16[3]             # Roll angle    (rad)
            p = x_f16[6]               # Roll rate     (rad/sec)
            theta = x_f16[4]           # Pitch angle   (rad)
            alpha = x_f16[1]           # AoA           (rad)

            if t_count == 200:
                CurrentState = self.STATE_ROLL

            elif CurrentState == self.STATE_ROLL:
                radsFromWingsLevel = round(phi/pi)
                # Until wings are "level" & roll rate is small
                if abs(phi - pi * radsFromWingsLevel) < eps_phi and abs(p) < eps_p:
                    CurrentState = self.STATE_PULL
                    self.sess.run(self.pull_state_assign,
                                  feed_dict={self.pull_state_placeholder: t_count*self.sim_step})

            elif CurrentState == self.STATE_PULL:
                radsFromNoseLevel = round((theta - alpha) / (2 * pi))
                if (theta - alpha) - 2 * pi * radsFromNoseLevel > path_goal:
                    CurrentState = self.STATE_DONE
                    self.sess.run(self.done_state_assign,
                                  feed_dict={self.done_state_placeholder: t_count*self.sim_step})

            elif CurrentState == self.STATE_DONE:
                #break
                pass

            t_count += 1


    def get_u_ref_tf(self, x_f16, t):
        with tf.name_scope("u_ref"):
            '''for the current discrete state, get the reference inputs signals'''
            # Pull out important variables for ease of use
            phi = x_f16[3]             # Roll angle    (rad)
            p = x_f16[6]               # Roll rate     (rad/sec)
            # q = x_f16[7]               # Pitch rate    (rad/sec)
            theta = x_f16[4]           # Pitch angle   (rad)
            alpha = x_f16[1]           # AoA           (rad)
            # Note: pathAngle = theta - alpha

            Nz_des = min(5, self.env.env.ctrlLimits.NzMax) # Desired maneuver g's

            def start_func():
                Nz = tf.constant(0.0, name="Nz", dtype=tf.float32)
                ps = tf.constant(0.0, name="ps", dtype=tf.float32)
                Ny_r = tf.constant(0.0, name="Ny_r", dtype=tf.float32)

                return Nz, ps, Ny_r

            def roll_func():
                radsFromWingsLevel = tf.round(phi/pi)
                K_prop = 4
                K_der = K_prop * 0.3

                Nz = tf.constant(0.0, name="Nz", dtype=tf.float32)
                ps = -(phi - pi * radsFromWingsLevel) * K_prop - p * K_der
                ps = tf.identity(ps, name="ps")
                Ny_r = tf.constant(0.0, name="Ny_r", dtype=tf.float32)

                return Nz, ps, Ny_r

            def pull_func():
                Nz = tf.constant(Nz_des, name="Nz", dtype=tf.float32)
                ps = tf.constant(0.0, name="ps", dtype=tf.float32)
                Ny_r = tf.constant(0.0, name="Ny_r", dtype=tf.float32)

                return Nz, ps, Ny_r

            def done_func():
                K_prop = 1
                K_der = K_prop*0.3
                radsFromWingsLevel = tf.round(phi/pi)
                ps = -(phi-pi*radsFromWingsLevel)*K_prop - p*K_der
                ps = tf.identity(ps, name="ps")

                K_prop2 = 2
                K_der2 = K_prop*0.3
                radsFromNoseLevel = tf.round((theta-alpha)/pi)
                Nz = -(theta - alpha - pi*radsFromNoseLevel) * K_prop2 - p*K_der2
                Nz = tf.identity(Nz, name="Nz")

                Ny_r = tf.constant(0.0, name="Ny_r", dtype=tf.float32)

                return Nz, ps, Ny_r

            error_composation = 1e-4
            Nz, ps, Ny_r = tf.case({tf.less(t, self.change_points[0]+error_composation): start_func,
                                    tf.logical_and(tf.greater(t, self.change_points[0]+error_composation), tf.less(t, self.change_points[1]+error_composation)): roll_func,
                                    tf.logical_and(tf.greater(t, self.change_points[1]+error_composation), tf.less(t, self.change_points[2]+error_composation)): pull_func},
                                    default=done_func, exclusive=True, strict=True, name="case")

            # basic speed control
            K_vt = 0.25
            throttle = -K_vt * (x_f16[0] - self.env.env.xequil[0])
            throttle = tf.identity(throttle, name="throttle")

            return Nz, ps, Ny_r, throttle

    def build_transformation(self, x_f16, u_deg, u_ref, t, multipliers):
        with tf.name_scope("transformation"):
            x_delta = x_f16 - self.env.env.stable_state
            x_ctrl = tf.gather_nd(x_delta, [[1], [7], [13], [2], [6], [8], [14], [15]])

            xd_model, Nz, Ny, _, _ = self.subf16_model_tf(x_f16[0:13], u_deg, multipliers)

            ps = x_ctrl[4] * tf.cos(x_ctrl[0]) + x_ctrl[5] * tf.sin(x_ctrl[0])

            Ny_r = Ny + x_ctrl[5]
            xd_integ = tf.stack([Nz - u_ref[0], ps - u_ref[1], Ny_r - u_ref[2]])

            xd = tf.concat([xd_model, xd_integ], 0)

            return x_f16+self.sim_step*xd

    def opt_reward_func(self, xt):
        # return -(tf.abs(xt[0]-center[0]) + tf.abs(xt[1]-center[1]) \
        #     + tf.abs(xt[2]-center[2]) + tf.abs(xt[11]-center[3]) + tf.abs(xt[13]-center[4]))
        # return tf.abs(xt)
        return -(tf.abs((xt[0]-self.env.env.center[0])/self.env.env.safe_norm_range[0])
        +tf.abs((xt[1]-self.env.env.center[1])/self.env.env.safe_norm_range[1])
        +tf.abs((xt[2]-self.env.env.center[2])/self.env.env.safe_norm_range[2])
        +tf.abs((xt[11]-self.env.env.center[3])/self.env.env.safe_norm_range[3])
        +tf.abs((xt[13]-self.env.env.center[4])/self.env.env.safe_norm_range[4]))

        # return -tf.reduce_sum((tf.gather(xt, [0, 1, 2, 11, 13]) - self.env.env.center)/self.env.env.safe_norm_range)

    @timeit
    def build_chain(self):
        """
            ONLY SUPPORT FOR MORELII NOW
        """

        with tf.name_scope("initial"):
            x0 = np.zeros([16], dtype=np.float32)
            self.grid_low = tf.placeholder(shape=x0.shape, dtype=tf.float32, name="grid_low")
            self.grid_high = tf.placeholder(shape=x0.shape, dtype=tf.float32, name="grid_high")
            self.x0 = tf.get_variable("x0", initializer=x0, dtype=tf.float32, trainable=True,
                                      constraint=lambda x: tf.clip_by_value(x, self.grid_low, self.grid_high))
            self.sess.run(tf.variables_initializer([self.x0]))
            xt = self.x0
            self.xt_list.append(xt)

            xcg_mult = 1.0 # center of gravity multiplier
            val = 1.0      # other aerodynmic coefficient multipliers
            cxt_mult = val
            cyt_mult = val
            czt_mult = val
            clt_mult = val
            cmt_mult = val
            cnt_mult = val
            multipliers = (xcg_mult, cxt_mult, cyt_mult, czt_mult, clt_mult, cmt_mult, cnt_mult)
            multipliers_tf = tf.constant(multipliers, dtype=tf.float32)

        # for i in range(self.step):
        #     t = tf.constant(i*self.sim_step, name="timestep", dtype=tf.float32)
        #     u_ref_tf = self.get_u_ref_tf(xt, t)
        #     u_deg_tf = self.build_neural_controller(xt, u_ref_tf)
        #     xt = self.build_transformation(xt, u_deg_tf, u_ref_tf, t, multipliers_tf)
        #     self.xt_list.append(xt)

        def cond(t, xt):
            return tf.less(t, self.step)

        def body(t, xt):
            t = tf.add(t, 1.0, name="timestep")
            u_ref_tf = self.get_u_ref_tf(xt, t*self.sim_step)
            u_deg_tf = self.build_neural_controller(xt, u_ref_tf)
            xt = self.build_transformation(xt, u_deg_tf, u_ref_tf, t, multipliers_tf)

            return t, xt

        t0 = tf.constant(0.0, dtype=tf.float32)
        t_final, out = tf.while_loop(cond, body, [t0, xt], parallel_iterations=10)

        self.xt_list.append(out)

        self.opt = tf.train.AdamOptimizer(learning_rate=self.opt_learning_rate)
        # self.cost_func = self.opt_reward_func(out)
        self.cost_func = self.opt_reward_func(self.xt_list[-1])

        # mask = [False, False, True, False, False, False, True, True, True, True, True, False, False, True, True, True]
        # update_var_list = []
        # count = 0
        # for p in mask:
        #     if not p:
        #         update_var_list.append(self.x0[count])
        #     count += 1
        # update_var = tf.stack(update_var_list)

        # print(update_var_list)
        # self.compute_gradients = self.opt.compute_gradients(self.cost_func, var_list=[self.x0])
        self.minimize_tf = self.opt.minimize(self.cost_func, var_list=[self.x0])

    @timeit
    def run(self, low, high, x0=None, iteration=20):
        if x0 is None:
            x0 = (low + high)/2

        assign_x0 = tf.assign(self.x0, x0)
        self.sess.run(assign_x0)

        previous_x0 = np.zeros(x0.shape, dtype=np.float32)
        x0_range = np.abs(high - low)

        self.update_change_points()

        # print("low:", low)
        # print("high:", high)
        for i in range(iteration):
            # ret = self.sess.run(self.compute_gradients, feed_dict={self.grid_low: low, self.grid_high: high})
            # print("gradient:", ret[0][0])
            self.sess.run(self.minimize_tf, feed_dict={self.grid_low: low, self.grid_high: high})
            # self.update_change_points()

            ret = self.sess.run([self.x0, self.xt_list[-1], self.cost_func])
            x0_ret, xt_ret, cost = ret
            # print("xt_ret:", xt_ret)
            # print("cost:", cost)

            diff = x0_ret - previous_x0

            if (previous_x0 == np.array(0.0, dtype=np.float32)).all():
                previous_x0.fill(1e-5)
            nonzero_index = diff.nonzero()

            change_rate_matrix = np.abs(diff[nonzero_index]/x0_range[nonzero_index])
            change_rate_sum = np.sum(change_rate_matrix)
            previous_x0 = x0_ret
            if change_rate_sum < 1e-4*len(x0):
                break
        print("converage at step", i)

        ret = self.sess.run([self.x0, self.xt_list[-1], self.cost_func])

        return ret

    def __del__(self):
        self.sess.close()


STATE_START = 'Waiting'
STATE_ROLL = 'Roll'
STATE_PULL = 'Pull'
STATE_DONE = 'Finished'

## unit test ##
def test_u_ref_tf():
    f16_model = F16TFModel(2000)
    f16_model.sess.run([f16_model.roll_state_assign, f16_model.pull_state_assign, f16_model.done_state_assign],
                                feed_dict={f16_model.roll_state_placeholder: 2.00-.001,
                                          f16_model.pull_state_placeholder: 3.27-.001,
                                          f16_model.done_state_placeholder: 9.98-.001})

    def template(t, x_16, state):
        x_f16_tf = tf.Variable(x_16, dtype=tf.float32)
        print(t)
        t_tf = tf.constant(t, name="timestep", dtype=tf.float32)
        f16_model.sess.run(tf.variables_initializer([x_f16_tf]))
        u_ref_tf = f16_model.get_u_ref_tf(x_f16_tf, t_tf, f16_model.sess)
        u_ref_tf_arr = np.array(f16_model.sess.run(u_ref_tf))

        f16_model.env.env.ap.state = state
        u_ref_arr = np.array(f16_model.env.env.ap.get_u_ref(t, x_16))

        print("u_ref_arr", u_ref_arr)
        print("u_ref_tf_arr", u_ref_tf_arr)
        print("diff sum:", np.sum(np.abs(u_ref_tf_arr - u_ref_arr)))


    x_16 = np.array([10000000]*16, dtype=np.float32)

    data = [(1.99,x_16, STATE_START), (2.01,x_16,STATE_ROLL), (4.0,x_16, STATE_PULL), (10.0, x_16, STATE_DONE)]
    for t, x_16, state in data:
        template(t, x_16, state)

def test_build_neural_controller():
    f16_model = F16TFModel()

    def template(x_f16_arr):
        with tf.variable_scope("test_nn", reuse=tf.AUTO_REUSE):
            class CustomPolicy(FeedForwardPolicy):
                def __init__(self, *args, **kwargs):
                    super(CustomPolicy, self).__init__(*args, **kwargs,
                                                       layers=[128, 128],
                                                       layer_norm=False,
                                                       feature_extraction="mlp")
            DVenv = DummyVecEnv([lambda: f16_model.env])
            model = DDPG.load(ROOT+"/trained_models/TDRL/f16/ddpg/128_128", env=DVenv, policy=CustomPolicy)

            # with model.graph.as_default():
            #     train_writer = tf.summary.FileWriter('./neural_network_graph', model.sess.graph)


            # f16_model.sess.run([f16_model.roll_state_assign, f16_model.pull_state_assign, f16_model.done_state_assign],
            #                     feed_dict={f16_model.roll_state_placeholder: 2.00,
            #                               f16_model.pull_state_placeholder: 3.27,
            #                               f16_model.done_state_placeholder: 7.98})

            t = tf.constant(9.98, name="timestep", dtype=tf.float32)

            x_f16 = tf.get_variable("xt0", initializer=x_f16_arr, dtype=tf.float32)
            f16_model.sess.run(tf.variables_initializer([x_f16]))
            u_ref = f16_model.get_u_ref_tf(x_f16, t)
            x_delta = x_f16_arr - f16_model.env.env.stable_state
            x_ctrl_arr = x_delta[np.array([1, 7, 13, 2, 6, 8, 14, 15])]
            u_deg = f16_model.build_neural_controller(x_f16, u_ref)

            print("build: ", f16_model.sess.run(u_deg))
            print("ori: ", model.predict(x_ctrl_arr))

    TEST_STEP = 1
    for _ in range(TEST_STEP):
        x_f16_arr = f16_model.env.env.initial_space.sample()
        template(x_f16_arr)


def test_transformation():
    f16_model = F16TFModel()
    env = gym.make("F16GCAS-v0")

    f16_model.sess.run([f16_model.roll_state_assign, f16_model.pull_state_assign, f16_model.done_state_assign],
                        feed_dict={f16_model.roll_state_placeholder: 2.00-.001,
                                  f16_model.pull_state_placeholder: 3.27-.001,
                                  f16_model.done_state_placeholder: 9.98-.001})
    env.reset()
    x_f16_arr = env.env.states[-1]
    t = tf.constant(0.01, name="timestep", dtype=tf.float32)

    x_f16 = tf.get_variable("xt0", initializer=x_f16_arr, dtype=tf.float32)
    f16_model.sess.run(tf.variables_initializer([x_f16]))
    u_ref = f16_model.get_u_ref_tf(x_f16, t)

    u_deg = f16_model.build_neural_controller(x_f16, u_ref)

    xcg_mult = 1.0 # center of gravity multiplier
    val = 1.0      # other aerodynmic coefficient multipliers
    cxt_mult = val
    cyt_mult = val
    czt_mult = val
    clt_mult = val
    cmt_mult = val
    cnt_mult = val
    multipliers = (xcg_mult, cxt_mult, cyt_mult, czt_mult, clt_mult, cmt_mult, cnt_mult)
    multipliers_tf = tf.constant(multipliers, dtype=tf.float32)

    transformation = f16_model.build_transformation(x_f16, u_deg, u_ref, t, multipliers_tf)

    print(f16_model.sess.run(transformation))

    u_deg_arr = f16_model.sess.run(u_deg)
    env.step(u_deg_arr[1:])
    print(env.env.states[-1])

def test_build_chain():
    chain_length = 1000
    f16_model = F16TFModel(chain_length)
    # f16_model.sess.run([f16_model.roll_state_assign, f16_model.pull_state_assign, f16_model.done_state_assign],
    #                             feed_dict={f16_model.roll_state_placeholder: 2.00-.001,
    #                                       f16_model.pull_state_placeholder: 3.27-.001,
    #                                       f16_model.done_state_placeholder: 9.98-.001})

    env = gym.make("F16GCAS-v0")
    ob = env.reset()
    x0 = env.states[-1]

    class CustomPolicy(FeedForwardPolicy):
                def __init__(self, *args, **kwargs):
                    super(CustomPolicy, self).__init__(*args, **kwargs,
                                                       layers=[128, 128],
                                                       layer_norm=False,
                                                       feature_extraction="mlp")
    DVenv = DummyVecEnv([lambda: f16_model.env])
    model = DDPG.load(ROOT+"/trained_models/TDRL/f16/ddpg/128_128", env=DVenv, policy=CustomPolicy)

    assign_x0 = tf.assign(f16_model.x0, x0)
    f16_model.sess.run(assign_x0)
    f16_model.update_change_points()
    x_out = f16_model.sess.run(f16_model.xt_list[-1])

    trace, reward = simulation_with_nn(env, chain_length, model, x0, mute=True)

    print("")
    print(x_out - env.states[-1])

def test_update_change_point():
    f16_model = F16TFModel(10)
    print(f16_model.sess.run(f16_model.change_points))

def test_change_point():
    f16_model = F16TFModel(850)
    sim = np.array(f16_model.env.env.states[835:845])
    build = np.array(f16_model.sess.run(f16_model.xt_list[835:845]))
    diff = np.abs(sim - build)
    max_diff = np.max(diff)
    print(max_diff)


import time
def test_eff_one_step():
    f16_model = F16TFModel(1)
    env = gym.make("F16GCAS-v0")

    f16_model.sess.run([f16_model.roll_state_assign, f16_model.pull_state_assign, f16_model.done_state_assign],
                        feed_dict={f16_model.roll_state_placeholder: 2.00-.001,
                                  f16_model.pull_state_placeholder: 3.27-.001,
                                  f16_model.done_state_placeholder: 9.98-.001})
    env.reset()
    x_f16_arr = env.env.states[-1]
    t = tf.constant(0.01, name="timestep", dtype=tf.float32)

    x_f16 = tf.get_variable("xt0", initializer=x_f16_arr, dtype=tf.float32)
    f16_model.sess.run(tf.variables_initializer([x_f16]))
    u_ref = f16_model.get_u_ref_tf(x_f16, t)

    t0 = time.time()
    u_deg = f16_model.build_neural_controller(x_f16, u_ref)
    t1 = time.time()
    print("Build Neural Network time:", t1-t0)

    xcg_mult = 1.0 # center of gravity multiplier
    val = 1.0      # other aerodynmic coefficient multipliers
    cxt_mult = val
    cyt_mult = val
    czt_mult = val
    clt_mult = val
    cmt_mult = val
    cnt_mult = val
    multipliers = (xcg_mult, cxt_mult, cyt_mult, czt_mult, clt_mult, cmt_mult, cnt_mult)
    multipliers_tf = tf.constant(multipliers, dtype=tf.float32)

    t0 = time.time()
    x1 = f16_model.build_transformation(x_f16, u_deg, u_ref, t, multipliers_tf)
    t1 = time.time()
    print("Build transformation time:", t1-t0)

    t0 = time.time()
    f16_model.sess.run(x1)
    t1 = time.time()
    print("Run x1 time:", t1-t0)

    var_before = set(tf.global_variables())
    opt = tf.train.AdamOptimizer()
    t0 = time.time()
    mini_tf = opt.minimize(tf.reduce_sum(tf.abs(x1)), var_list=[x_f16[0]])
    t1 = time.time()
    print("x1 opt build time:", t1-t0)
    var_after = set(tf.global_variables())
    initializer = tf.variables_initializer(var_after-var_before)
    f16_model.sess.run(initializer)

    t0 = time.time()
    for _ in range(1000):
        f16_model.sess.run(mini_tf)
    t1 = time.time()
    print("x1 opt run time", t1-t0)

def test_chain_eff():
    f16_model5 = F16TFModel(1000)
    x0 = f16_model5.xt_list[0]
    x5 = f16_model5.xt_list[-1]

    var_before = set(tf.global_variables())
    opt = tf.train.AdamOptimizer()
    t0 = time.time()
    mini_tf = opt.minimize(tf.reduce_sum(tf.abs(x5)), var_list=[x0])
    t1 = time.time()
    print("x5 opt build time:", t1-t0)
    var_after = set(tf.global_variables())
    initializer = tf.variables_initializer(var_after-var_before)
    f16_model5.sess.run(initializer)

    t0 = time.time()
    for _ in range(10):
        f16_model5.sess.run(mini_tf)
    t1 = time.time()
    print("x5 opt run time", t1-t0)

def test_projected_gradient_descent():
    f16_model = F16TFModel(1000)
    low = np.array([ 4.34551666e+02,  2.97966916e-02,  0.00000000e+00,  6.32029772e-01,
                    -1.01124763e+00, -6.32029772e-01,  0.00000000e+00,  0.00000000e+00,
                     0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  2.89701099e+03,
                     7.24252748e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00])
    high = np.array([ 4.06981995e+02,  2.79062719e-02,  0.00000000e+00,  5.91931313e-01,
                    -9.47090097e-01, -5.91931313e-01,  0.00000000e+00,  0.00000000e+00,
                     0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  2.71321318e+03,
                     6.78303298e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00])
    x0 = (low + high)/2

    f16_model.reset_initial_bound(low, high, x0)
    f16_model.run_minimize(1)

def test_variable_cut():
    f16_model = F16TFModel(1000)
    low = np.array([ 4.34551666e+02,  2.97966916e-02,  0.00000000e+00,  6.32029772e-01,
                    -1.01124763e+00, -6.32029772e-01,  0.00000000e+00,  0.00000000e+00,
                     0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  2.89701099e+03,
                     7.24252748e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00], dtype=np.float32)
    high = np.array([ 4.06981995e+02,  2.79062719e-02,  0.00000000e+00,  5.91931313e-01,
                    -9.47090097e-01, -5.91931313e-01,  0.00000000e+00,  0.00000000e+00,
                     0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  2.71321318e+03,
                     6.78303298e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00], dtype=np.float32)
    x0 = (low + high)/2

    for _ in range(5):
        f16_model.reset(low, high, x0)
        print("minimize variables:", len(f16_model.graph.as_graph_def().node))
        f16_model.run(1)

def test_new_api():
    f16_model = F16TFModel(1000)
    low = np.array([ 4.34551666e+02,  2.97966916e-02,  0.00000000e+00,  6.32029772e-01,
                    -1.01124763e+00, -6.32029772e-01,  0.00000000e+00,  0.00000000e+00,
                     0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  2.89701099e+03,
                     7.24252748e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00], dtype=np.float32)
    high = np.array([ 4.06981995e+02,  2.79062719e-02,  0.00000000e+00,  5.91931313e-01,
                    -9.47090097e-01, -5.91931313e-01,  0.00000000e+00,  0.00000000e+00,
                     0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  2.71321318e+03,
                     6.78303298e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00], dtype=np.float32)
    x0 = (low + high)/2
    print(f16_model.run(low=low, high=high, x0=x0))

if __name__ == "__main__":
    # test_u_ref_tf()
    # test_build_neural_controller()
    # test_transformation()
    # test_build_chain()
    # test_update_change_point()
    # test_change_point()
    # test_eff_one_step()
    # test_chain_eff()
    # test_projected_gradient_descent()
    # test_variable_cut()
    test_new_api()
