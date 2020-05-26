# -*- coding: utf-8 -*-
# -------------------------------
# Author: Zikang Xiong
# Email: zikangxiong@gmail.com
# Date:   2019-06-19 09:40:01
# Last Modified by:   Zikang Xiong
# Last Modified time: 2019-11-11 16:53:20
# -------------------------------
import gym
import numpy as np
from gym import spaces

from math import pi
from numpy import deg2rad

from CoRec.envs.F16.AeroBenchVVPython.code.PassFailAutomaton import FlightLimitsPFA, FlightLimits
from CoRec.envs.F16.AeroBenchVVPython.code.CtrlLimits import CtrlLimits
from CoRec.envs.F16.AeroBenchVVPython.code.controlledF16 import F16ManullyControl
from CoRec.envs.F16.AeroBenchVVPython.code.Autopilot import GcasAutopilot

import tensorflow as tf
from TVTDRL.algorithms.NNTools.controller import get_stable_baselines_nn_weights, build_stable_baselines_controller

from tensorflow.keras.models import load_model
from tensorflow.keras import Model

import os
import CoRec
ROOT = os.path.dirname(os.path.abspath(CoRec.__file__))+"/envs/F16/AeroBenchVVPython/code/"

from tensorflow.keras import backend as K


# from CoRec.envs.F16.plot import plot3d_anim

class F16GCAS(gym.Env):

    def __init__(self):   
        # Initial condition
        self.power_low = 9
        self.power_high = 9
        # Default alpha & beta
        self.alpha_low = deg2rad(2.1215)
        self.alpha_high = deg2rad(2.1215)
        self.beta_low = 0
        self.beta_high = 0
        # Initial Attitude
        self.alt_low = 3600
        self.alt_high = 3600
        self.Vt_low  = 540                   
        self.Vt_high = 540                   # Pass at Vtg = 540;    Fail at Vtg = 550;
        self.phi_low  = (pi/2)*0.5              
        self.phi_high = (pi/2)*0.5           # Roll angle from wings level (rad)
        self.theta_low  = (-pi/2)*0.8
        self.theta_high = (-pi/2)*0.8        # Pitch angle from nose level (rad)
        self.psi_low = -pi/4
        self.psi_high = -pi/4                # Yaw angle from North (rad)
        # state = [VT, alpha, beta, phi, theta, psi, P, Q, R, pn, pe, h, pow, Nz, Ps, Ny]
        self.initial_state_low = np.array([self.Vt_high, self.alpha_high, self.beta_high, self.phi_high, self.theta_high, self.psi_high, 0, 0, 0, 0, 0, self.alt_high, self.power_high, 0, 0, 0])/1.10 #1.82
        self.initial_state_high = np.array([self.Vt_low, self.alpha_low, self.beta_low, self.phi_low, self.theta_low, self.psi_low, 0, 0, 0, 0, 0, self.alt_low, self.power_low, 0, 0, 0])*1.01 #1.06
        self.initial_space = spaces.Box(self.initial_state_low, self.initial_state_high, dtype=np.float32)

        # Safety Constrains
        self.flightLimits = FlightLimits()
        self.state_high = np.full(len(self.initial_state_low), np.inf)
        self.state_low = np.full(len(self.initial_state_low), np.NINF)
        self.state_low[0] = self.flightLimits.vMin
        self.state_low[1] = self.flightLimits.alphaMinDeg
        self.state_low[2] = -self.flightLimits.betaMaxDeg
        self.state_low[11] = self.flightLimits.altitudeMin
        self.state_low[13] = self.flightLimits.NzMin
        self.state_high[0] = self.flightLimits.vMax
        self.state_high[1] = self.flightLimits.alphaMaxDeg
        self.state_high[2] = self.flightLimits.betaMaxDeg
        self.state_high[11] = self.flightLimits.altitudeMax
        self.state_high[13] = self.flightLimits.NzMax
        self.original = (self.state_low + self.state_high) / 2
        self.center = ((self.state_low + self.state_high) / 2)[([0, 1, 2, 11, 13])]
        self.safe_space = spaces.Box(self.state_low, self.state_high, dtype=np.float32)              # Yaw angle from North (rad)
        self.safe_norm_range = (self.state_high - self.state_low)[([0, 1, 2, 11, 13])]

        ctrl_state_low = np.array([self.state_low[i] for i in [1, 7, 13, 2, 6, 8, 14, 15]], dtype=np.float32)
        ctrl_state_high = np.array([self.state_high[i] for i in [1, 7, 13, 2, 6, 8, 14, 15]], dtype=np.float32)
        self.observation_space = spaces.Box(ctrl_state_low, ctrl_state_high, dtype=np.float32)

        # control limits 
        self.ctrlLimits = CtrlLimits()
        self.u_low = np.array([self.ctrlLimits.ThrottleMin, self.ctrlLimits.ElevatorMinDeg, 
                        self.ctrlLimits.AileronMinDeg, self.ctrlLimits.RudderMinDeg])
        self.u_high = np.array([self.ctrlLimits.ThrottleMax, self.ctrlLimits.ElevatorMaxDeg, 
                        self.ctrlLimits.AileronMaxDeg, self.ctrlLimits.RudderMaxDeg])
        self.action_space = spaces.Box(self.u_low[1:4], self.u_high[1:4], dtype=np.float32)

        # Stable states        
        self.xequil = np.array([502.0, 0.03887505597600522, 0.0, 0.0, 0.03887505597600522, 0.0, 0.0, 0.0, \
                        0.0, 0.0, 0.0, 1000.0, 9.05666543872074], dtype=np.float32).transpose()
        self.uequil = np.array([0.13946204864060271, -0.7495784725828754, 0.0, 0.0], dtype=np.float32).transpose()
        self.stable_state = np.concatenate((self.xequil, [0.0, 0.0, 0.0]))
        self.stable_action = np.concatenate((self.uequil, [0.0, 0.0, 0.0]))

        # Select Desired F-16 Plant
        self.f16_plant = 'morelli' # 'stevens' or 'morelli'

        xcg_mult = 1.0 # center of gravity multiplier
        val = 1.0      # other aerodynmic coefficient multipliers
        cxt_mult = val
        cyt_mult = val
        czt_mult = val
        clt_mult = val
        cmt_mult = val
        cnt_mult = val
        self.multipliers = (xcg_mult, cxt_mult, cyt_mult, czt_mult, clt_mult, cmt_mult, cnt_mult)

        self.ap = GcasAutopilot(self.xequil, self.uequil, self.flightLimits, self.ctrlLimits)
        self.pass_fail = FlightLimitsPFA(self.flightLimits)

        self.sim_step = 0.01

        # Q: Penalty on State Error in LQR controller
        # These were chosen to try to achieve a natural frequency of 3 rad/sec and a damping ratio (zeta) of 0.707
        # see the matlab code for more analysis of the resultant controllers
        # state = [VT, alpha, beta, phi, theta, psi, P, Q, R, pn, pe, h, pow, Nz, Ps, Ny]
        q_alpha = 1000
        q_q = 0
        q_Nz = 1500
        q_beta = 0
        q_p = 0
        q_r = 0
        q_ps_i = 1200
        q_Ny_r_i = 3000
        q_list = [0, q_alpha, q_beta, 0, 0, 0, q_p, q_q, q_r, 0, 0, 0, 0, q_Nz, q_ps_i, q_Ny_r_i]
        self.Q = np.diag(q_list)

        # R: Penalty on Control Effort in LRQ controller
        r_elevator = 1
        r_aileron = 1
        r_rudder = 1
        r_list = [0, r_elevator, r_aileron, r_rudder, 0, 0, 0]
        self.R = np.diag(r_list)

        self.ut = None
        self.xdt = None

        self.reward_sum = 0.0

        self.viewer = None

        self.A = np.array([[-1.85458009e-02, 7.64493830e+00, -7.78917511e-06, -3.12336055e-06, -3.21700000e+01, 0.00000000e+00, 0.00000000e+00, -6.16876227e-01, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.95007313e-05, 3.92124843e-01],
            [-2.53750397e-04, -9.86488259e-01, 3.27654085e-06, -1.59967156e-06, -1.24562802e-05, 0.00000000e+00, 0.00000000e+00, 9.07620613e-01, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.87658865e-06, -3.03815918e-05],
            [0.00000000e+00, 0.00000000e+00, -3.12746139e-01, 6.40349808e-02, 0.00000000e+00, 0.00000000e+00, 3.84147116e-02, 0.00000000e+00, -9.91802736e-01, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
            [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00, 0.00000000e+00, 3.88946514e-02, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
            [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
            [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00075611e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
            [0.00000000e+00, 0.00000000e+00, -3.00508220e+01, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, -3.56844338e+00, 2.62720000e-04, 6.54741930e-01, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
            [-7.98036222e-14, 7.98739107e-01, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, -8.79500000e-06, -1.04652744e+00, -2.85840500e-03, 0.00000000e+00, 0.00000000e+00, 5.84119263e-16, 0.00000000e+00],
            [0.00000000e+00, 0.00000000e+00, 8.29995934e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, -2.64250112e-02, 2.53920000e-03, -4.62739115e-01, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
            [1.00000000e+00, -8.77324816e-06, 0.00000000e+00, -1.89857019e-06, -8.77324816e-06, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
            [0.00000000e+00, 0.00000000e+00, 5.02000000e+02, -1.95103622e+01, 0.00000000e+00, 5.02000000e+02, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
            [-7.07711888e-16, -5.02000000e+02, 0.00000000e+00, 4.87432317e-06, 5.02000000e+02, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
            [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, -1.00000000e+00]])
    
        self.B = np.array([[0.00000000e+00, 1.66130880e-01, 0.00000000e+00, 0.00000000e+00],
                    [0.00000000e+00, -2.08865123e-03, 0.00000000e+00, 0.00000000e+00],
                    [0.00000000e+00, -0.00000000e+00, 2.86568650e-04, 7.82377902e-04],
                    [0.00000000e+00, -0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
                    [0.00000000e+00, -0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
                    [0.00000000e+00, -0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
                    [0.00000000e+00, -0.00000000e+00, -7.12504959e-01, 1.27548586e-01],
                    [0.00000000e+00, -1.70464357e-01, 0.00000000e+00, 0.00000000e+00],
                    [0.00000000e+00, -0.00000000e+00, -3.09056326e-02, -6.02352593e-02],
                    [0.00000000e+00, -0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
                    [0.00000000e+00, -0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
                    [0.00000000e+00, -0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
                    [6.49400000e+01, -0.00000000e+00, 0.00000000e+00, 0.00000000e+00]])

        # self.pre_ap_state = self.ap.state

    def step(self, ut):
        u_ref = self.ap.get_u_ref(self.times[-1], self.states[-1])

        u_deg = np.zeros((4,))
        u_deg[1:4] = ut
        u_deg[0] = u_ref[3]
        u_deg[0:4] += self.uequil
        u_deg = np.clip(u_deg, self.u_low, self.u_high)

        self.ut = u_deg        
        xd, u, Nz, ps, Ny_r = F16ManullyControl(self.times[-1], self.states[-1], self.f16_plant, self.ap, \
                              self.x_ctrl, u_deg, multipliers=self.multipliers)
        self.xdt = xd
        # xd_linear = self.states[-1][:13].dot(self.A.T) + u_deg.dot(self.B.T)
        # print("diff xd between xd_liear", xd[:13] - xd_linear)
        self.pass_fail.advance(self.times[-1], self.states[-1], self.ap.state, xd, u, Nz, ps, Ny_r)
        self.Nz_list.append(Nz)
        self.ps_list.append(ps)
        self.u_list.append(u)

        t = self.times[-1] + self.sim_step
        state = self.states[-1]+self.sim_step*xd
        state = state.astype(np.float32)
        # print("1 state type:", type(state[0]))
        # print("len(env.states):", len(self.states))
        self.times.append(t)
        self.states.append(state)
        self.state = state
        # assert False

        updated = self.ap.advance_discrete_state(self.times[-1], self.states[-1])
        self.modes.append(self.ap.state)
        if updated:
            pass
            #print("at time {}, state changes to {}".format(self.times[-1], self.ap.state))

        done = (not self.pass_fail.result()) and self.pass_fail.break_on_error
        if not done:
            reward = self.reward_func(self.states[-1], self.u_list[-1])
            self.reward_sum += reward
        else:
            reward = -1.0 - self.reward_sum

        return self.x_ctrl, reward, done, {}

    def reward_func(self, state, action):
        # return -(np.sum(np.dot(self.Q, np.abs(state-self.stable_state)))\
        #         +np.sum(np.dot(self.R, np.abs(action-self.stable_action))))*1e-7
        return -np.sum(np.abs(state[([0, 1, 2, 11, 13])] - self.center)/self.safe_norm_range)*1e-5

    def reset(self, x0=None):
        if x0 is None:
            x0 = self.initial_space.sample()
        x0 = x0.astype(np.float32)

        self.pass_fail.reset()
        self.ap.reset()

        assert type(x0) is np.ndarray

        # run the numerical simulation
        self.times = [0.0]
        self.states = [x0]
        self.state = x0
        self.modes = [self.ap.state]

        self.Nz_list = []
        self.ps_list = []
        self.u_list = []

        x0_copy = x0.copy()
        x0_copy[:len(self.xequil)] -= self.xequil

        self.reward_sum = 0.0

        return x0_copy[[1, 7, 13, 2, 6, 8, 14, 15]]

    @property    
    def x_ctrl(self):
        # Calculate perturbation from trim state
        x_delta = self.states[-1].copy()
        x_delta[:len(self.xequil)] -= self.xequil
        return np.array([x_delta[i] for i in [1, 7, 13, 2, 6, 8, 14, 15]], dtype=np.float32)

    def check_in_initial(self, x0):
        pass

    def check_safe(self, xt):
        assert xt.shape == self.states[-1].shape
        return self.safe_space.contains(xt)

    def transformation_function(self, xt_tf, ut_tf):
        with tf.name_scope("transformation_function"):
            xt_tf = xt_tf + self.vrlenv.timestep * (tf.tensordot(xt_tf, tf.transpose(self.A_tf), axes=1)
                                                    + tf.tensordot(ut_tf, tf.transpose(self.B_tf), axes=1))
            return xt_tf

    def get_weights(self, nn_model):
        self.w_list, self.b_list = get_stable_baselines_nn_weights(nn_model)

    def build_controller(self, xt):
        with tf.name_scope("controller"):
            ut_tf = build_stable_baselines_controller(xt, self.u_max, self.w_list, self.b_list)
        return ut_tf

    def approximator(self, x0, algo, step=2000):
        if step == 2000:
            approx = load_model(ROOT+"/nn_model/trained_model/approx_1000_2000.model")

        new_input = tf.keras.Input(tensor=tf.reshape(x0, (1,16)))
        new_output = approx(new_input)
        new_model = Model(new_input, new_output)

        sess = K.get_session()

        return tf.reshape(new_model.output, (16, )), sess

    def render(self, mode='human'):
        if self.viewer is None:
            pass

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
