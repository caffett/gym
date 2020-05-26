# -*- coding: utf-8 -*-
# -------------------------------
# Author: Zikang Xiong
# Email: zikangxiong@gmail.com
# Date:   2019-06-20 22:00:42
# Last Modified by:   Zikang Xiong
# Last Modified time: 2019-07-05 01:11:24
# -------------------------------
import numpy as np
from numpy import deg2rad
from math import pi

from RunF16Sim import RunF16Sim, RunF16Sim_discete_time
from PassFailAutomaton import AirspeedPFA, FlightLimitsPFA, FlightLimits
from CtrlLimits import CtrlLimits
from LowLevelController import LowLevelController, NNLowLevelController
from Autopilot import FixedSpeedAutopilot, GcasAutopilot
from controlledF16 import controlledF16, F16ManullyControl
from BuildNNController import NeuralNetworkContorller

from plot import plot2d, plot3d_anim

import gym 
import CoRec

def step_test():
    ctrlLimits = CtrlLimits()
    flightLimits = FlightLimits()
    llc = LowLevelController(ctrlLimits)

    setpoint = 2220
    p_gain = 0.01

    ap = FixedSpeedAutopilot(setpoint, p_gain, llc.xequil, llc.uequil, flightLimits, ctrlLimits)

    # If the airspeed can get 2220 in 60s with 5% error
    pass_fail = AirspeedPFA(60, setpoint, 5)

    ### Initial Conditions ###
    power = 0 # Power

    # Default alpha & beta
    alpha = deg2rad(2.1215) # Trim Angle of Attack (rad)
    beta = 0                # Side slip angle (rad)

    alt = 20000 # Initial Attitude
    Vt = 1000 # Initial Speed
    phi = 0 #(pi/2)*0.5           # Roll angle from wings level (rad)
    theta = 0 #(-pi/2)*0.8        # Pitch angle from nose level (rad)
    psi = 0 #-pi/4                # Yaw angle from North (rad)

    # Build Initial Condition Vectors
    # state = [VT, alpha, beta, phi, theta, psi, P, Q, R, pn, pe, h, pow]
    initialState = [Vt, alpha, beta, phi, theta, psi, 0, 0, 0, 0, 0, alt, power]

    # Select Desired F-16 Plant
    f16_plant = 'morelli' # 'stevens' or 'morelli'

    tMax = 70 # simulation time

    def der_func1(t, y):
        'derivative function'

        der = controlledF16(t, y, f16_plant, ap, llc)[0]

        rv = np.zeros((y.shape[0],))

        rv[0] = der[0] # speed
        rv[12] = der[12] # power lag term

        return rv

    def der_func2(t, y):
        """
            manually controled F16
        """
        der = F16ManullyControl(t, y, f16_plant, ap, llc)[0]

        rv = np.zeros((y.shape[0],))

        rv[0] = der[0] # speed
        rv[12] = der[12] # power lag term

        return rv

    passed, times, states, modes, ps_list, Nz_list, u_list = \
        RunF16Sim(initialState, tMax, der_func1, f16_plant, ap, llc, pass_fail, sim_step=0.1)
    print(states)
    print("Simulation Conditions Passed: {}".format(passed))

    # plot
    filename = "engine_e1.png" # engine_e.png
    plot2d(filename, times, [(states, [(0, 'Vt'), (12, 'Pow')]), (u_list, [(0, 'Throttle')])])

    passed, times, states, modes, ps_list, Nz_list, u_list = \
        RunF16Sim(initialState, tMax, der_func2, f16_plant, ap, llc, pass_fail, sim_step=0.1)
    print(states)
    print("Simulation Conditions Passed: {}".format(passed))

    # plot
    filename = "engine_e2.png" # engine_e.png
    plot2d(filename, times, [(states, [(0, 'Vt'), (12, 'Pow')]), (u_list, [(0, 'Throttle')])])


def single_step_control_logic():
    ctrlLimits = CtrlLimits()
    flightLimits = FlightLimits()

    # If the airspeed can get 2220 in 60s with 5% error
    pass_fail = AirspeedPFA(60, setpoint, 5)

    ### Initial Conditions ###
    power = 0 # Power

    # Default alpha & beta
    alpha = deg2rad(2.1215) # Trim Angle of Attack (rad)
    beta = 0                # Side slip angle (rad)

    alt = 20000 # Initial Attitude
    Vt = 1000 # Initial Speed
    phi = 0 #(pi/2)*0.5           # Roll angle from wings level (rad)
    theta = 0 #(-pi/2)*0.8        # Pitch angle from nose level (rad)
    psi = 0 #-pi/4                # Yaw angle from North (rad)

    # Build Initial Condition Vectors
    # state = [VT, alpha, beta, phi, theta, psi, P, Q, R, pn, pe, h, pow]
    initialState = [Vt, alpha, beta, phi, theta, psi, 0, 0, 0, 0, 0, alt, power]

    # Build Initial Condition Vectors
    # state = [VT, alpha, beta, phi, theta, psi, P, Q, R, pn, pe, h, pow]
    initialState = [Vt, alpha, beta, phi, theta, psi, 0, 0, 0, 0, 0, alt, power]

    # Select Desired F-16 Plant
    f16_plant = 'morelli' # 'stevens' or 'morelli'

    tMax = 70 # simulation time

    # air plane model
    setpoint = 2220
    p_gain = 0.01
    ap = FixedSpeedAutopilot(setpoint, p_gain, None, None, flightLimits, ctrlLimits)
    
def trim_point_test():

    flightLimits = FlightLimits()
    ctrlLimits = CtrlLimits()
    llc = LowLevelController(ctrlLimits)
    ap = GcasAutopilot(llc.xequil, llc.uequil, flightLimits, ctrlLimits)
    pass_fail = FlightLimitsPFA(flightLimits)
    pass_fail.break_on_error = False

    # ### Initial Conditions ###
    # initialState = [Vt, alpha, beta, phi, theta, psi, 0, 0, 0, 0, 0, alt, power]
    initialState = [502.0, 0.03887505597600522, 0.0, 0.0, 0.03887505597600522, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1000.0, 9.05666543872074]

    # if not None will do animation. Try a filename ending in .gif or .mp4 (slow). Using '' will plot to the screen.
    animFilename = 'gcas_stable.gif'

    # Select Desired F-16 Plant
    f16_plant = 'morelli' # 'stevens' or 'morelli'

    tMax = 15 # simulation time

    xcg_mult = 1.0 # center of gravity multiplier

    val = 1.0      # other aerodynmic coefficient multipliers
    cxt_mult = val
    cyt_mult = val
    czt_mult = val
    clt_mult = val
    cmt_mult = val
    cnt_mult = val

    multipliers = (xcg_mult, cxt_mult, cyt_mult, czt_mult, clt_mult, cmt_mult, cnt_mult)

    der_func = lambda t, y: controlledF16(t, y, f16_plant, ap, llc, multipliers=multipliers)[0]

    passed, times, states, modes, ps_list, Nz_list, u_list = RunF16Sim(\
        initialState, tMax, der_func, f16_plant, ap, llc, pass_fail, multipliers=multipliers)

    print("Simulation Conditions Passed: {}".format(passed))

    if animFilename is not None:
        plot3d_anim(times, states, modes, ps_list, Nz_list, filename=animFilename)

def neural_network_controller_test():

    flightLimits = FlightLimits()
    ctrlLimits = CtrlLimits()
    nn_model = NeuralNetworkContorller()
    llc = NNLowLevelController(ctrlLimits, nn_model)
    llc = LowLevelController(ctrlLimits)
    ap = GcasAutopilot(llc.xequil, llc.uequil, flightLimits, ctrlLimits)
    pass_fail = FlightLimitsPFA(flightLimits)
    pass_fail.break_on_error = False

    ### Initial Conditions ###
    power = 9 # Power

    # Default alpha & beta
    alpha = deg2rad(2.1215) # Trim Angle of Attack (rad)
    beta = 0                # Side slip angle (rad)

    # Initial Attitude
    alt = 3600
    Vt = 540                   # Pass at Vtg = 540;    Fail at Vtg = 550;
    phi = (pi/2)*0.5           # Roll angle from wings level (rad)
    theta = (-pi/2)*0.8        # Pitch angle from nose level (rad)
    psi = -pi/4                # Yaw angle from North (rad)

    # Build Initial Condition Vectors
    # state = [VT, alpha, beta, phi, theta, psi, P, Q, R, pn, pe, h, pow]
    initialState = [Vt, alpha, beta, phi, theta, psi, 0, 0, 0, 0, 0, alt, power]

    # if not None will do animation. Try a filename ending in .gif or .mp4 (slow). Using '' will plot to the screen.
    animFilename = 'gcas_dis.gif'

    # Select Desired F-16 Plant
    f16_plant = 'morelli' # 'stevens' or 'morelli'

    tMax = 15 # simulation time

    xcg_mult = 1.0 # center of gravity multiplier

    val = 1.0      # other aerodynmic coefficient multipliers
    cxt_mult = val
    cyt_mult = val
    czt_mult = val
    clt_mult = val
    cmt_mult = val
    cnt_mult = val

    multipliers = (xcg_mult, cxt_mult, cyt_mult, czt_mult, clt_mult, cmt_mult, cnt_mult)

    der_func = lambda t, y: controlledF16(t, y, f16_plant, ap, llc, multipliers=multipliers)[0]

    # passed, times, states, modes, ps_list, Nz_list, u_list = RunF16Sim(\
    #     initialState, tMax, der_func, f16_plant, ap, llc, pass_fail, multipliers=multipliers)
    print("!!!")
    passed, times, states, modes, ps_list, Nz_list, u_list = RunF16Sim_discete_time(\
        initialState, tMax, f16_plant, ap, llc, pass_fail, multipliers=multipliers)
    print("!!!!")

    print("Simulation Conditions Passed: {}".format(passed))

    # if animFilename is not None:
    #     plot3d_anim(times, states, modes, ps_list, Nz_list, filename=animFilename)

from gym import spaces
class GymInterface:
    def __init__(self):   
        # Initial condition
        self.power_low = 9
        self.power_high = 9
        # Default alpha & beta
        self.alpha_low = deg2rad(21.215)
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
        self.state_low = np.array([self.Vt_low, self.alpha_low, self.beta_low, self.phi_low, self.theta_low, self.psi_low, 0, 0, 0, 0, 0, self.alt_low, self.power_low, 0, 0, 0])
        self.state_high = np.array([self.Vt_high, self.alpha_high, self.beta_high, self.phi_high, self.theta_high, self.psi_high, 0, 0, 0, 0, 0, self.alt_high, self.power_high, 0, 0, 0])
        self.initial_space = spaces.Box(self.state_low, self.state_high, dtype=float)

        # Safety Constrains
        self.flightLimits = FlightLimits()
        state_high = np.full(len(self.state_low), np.inf)
        state_low = np.full(len(self.state_low), np.NINF)
        state_low[0] = self.flightLimits.vMin
        state_low[1] = self.flightLimits.alphaMinDeg
        state_low[2] = -self.flightLimits.betaMaxDeg
        state_low[11] = self.flightLimits.altitudeMin
        state_high[0] = self.flightLimits.vMax
        state_high[1] = self.flightLimits.alphaMaxDeg
        state_high[2] = self.flightLimits.betaMaxDeg
        state_high[11] = self.flightLimits.altitudeMax
        self.observation_space = spaces.Box(state_low, state_high, dtype=float)              # Yaw angle from North (rad)

        # control limits 
        self.ctrlLimits = CtrlLimits()
        self.u_low = np.array([self.ctrlLimits.ThrottleMin, self.ctrlLimits.ElevatorMinDeg, 
                        self.ctrlLimits.AileronMinDeg, self.ctrlLimits.RudderMinDeg])
        self.u_high = np.array([self.ctrlLimits.ThrottleMax, self.ctrlLimits.ElevatorMaxDeg, 
                        self.ctrlLimits.AileronMaxDeg, self.ctrlLimits.RudderMaxDeg])

        # Stable states        
        self.xequil = np.array([502.0, 0.03887505597600522, 0.0, 0.0, 0.03887505597600522, 0.0, 0.0, 0.0, \
                        0.0, 0.0, 0.0, 1000.0, 9.05666543872074], dtype=float).transpose()
        self.uequil = np.array([0.13946204864060271, -0.7495784725828754, 0.0, 0.0], dtype=float).transpose()
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

        self.viewer = None

    def step(self, ut):
        u_ref = self.ap.get_u_ref(self.times[-1], self.states[-1])

        u_deg = np.zeros((4,))
        u_deg[1:4] = ut
        u_deg[0] = u_ref[3]
        u_deg[0:4] += self.uequil
        u_deg = np.clip(u_deg, self.u_low, self.u_high)
        
        xd, u, Nz, ps, Ny_r = F16ManullyControl(self.times[-1], self.states[-1], self.f16_plant, self.ap, \
                              self.x_ctrl, u_deg, multipliers=self.multipliers)
        self.pass_fail.advance(self.times[-1], self.states[-1], self.ap.state, xd, u, Nz, ps, Ny_r)
        self.Nz_list.append(Nz)
        self.ps_list.append(ps)
        self.u_list.append(u)

        t = self.times[-1] + self.sim_step
        state = self.states[-1]+self.sim_step*xd
        self.times.append(t)
        self.states.append(state)

        updated = self.ap.advance_discrete_state(self.times[-1], self.states[-1])
        self.modes.append(self.ap.state)
        if updated:
            print("at time {}, state changes to {}".format(self.times[-1], self.ap.state))

        done = (not self.pass_fail.result()) and self.pass_fail.break_on_error
        if not done:
            reward = self.reward_func(self.states[-1], self.u_list[-1])
        else:
            reward = -1.0
        return self.states[-1], reward, done, {}

    def reward_func(self, state, action):
        return -(np.sum(np.dot(self.Q, np.abs(state-self.stable_state)))\
                +np.sum(np.dot(self.R, np.abs(action-self.stable_action))))*1e-7
    
    @property    
    def x_ctrl(self):
        # Calculate perturbation from trim state
        x_delta = self.states[-1].copy()
        x_delta[:len(self.xequil)] -= self.xequil
        return np.array([x_delta[i] for i in [1, 7, 13, 2, 6, 8, 14, 15]], dtype=float)

    def reset(self, x0=None):
        if x0 is None:
            x0 = self.initial_space.sample()

        assert type(x0) is np.ndarray

        # run the numerical simulation
        self.times = [0.0]
        self.states = [x0]
        self.modes = [self.ap.state]

        self.Nz_list = []
        self.ps_list = []
        self.u_list = []

        return x0

def GymEnvInterfaceTest():
    #env = GymInterface()
    env = gym.make("F16GCAS-v0")
    env.reset()
    # Longitudinal Gains
    K_lqr_long = np.array([[-156.8801506723475, -31.037008068526642, -38.72983346216317]], dtype=float)
    # Lateral Gains
    K_lqr_lat = np.array([[30.511411060051355, -5.705403676148551, -9.310178739319714, \
                                                -33.97951344944365, -10.652777306717681], \
                          [-22.65901530645282, 1.3193739204719577, -14.2051751789712, \
                                                6.7374079391328845, -53.726328142239225]], dtype=float)
    K_lqr = np.zeros((3, 8))
    K_lqr[:1, :3] = K_lqr_long
    K_lqr[1:, 3:] = K_lqr_lat

    reward_list = []

    done = False
    while not done:
        u = np.dot(-K_lqr, env.x_ctrl)
        # print(env.x_ctrl)
        # print(u)
        # print("!!!!")
        _, reward, done, _ = env.step(u)
        reward_list.append([reward])
        if done:
            break

    # from TVTDRL.metrics.draw import draw_state_list
    # draw_state_list(reward_list)

    # print(reward_list)
    # print("len env.times:", len(env.times))
    # print("len env.states:", len(env.states))
    # print("len env.modes:", len(env.modes))
    # print("len env.ps_list:", len(env.ps_list))
    # print("len env.Nz_list:", len(env.Nz_list))

    # from TVTDRL.metrics.draw import draw_state_list
    # for i in range(16):
    #     draw_state_list(env.states, state_index=[i])
        #draw_state_list(env.u_list, state_index=[i])
    
    #draw_state_list(env.states, state_index=[13, 14, 15])
    # print(env.states[-1])
    # print(env.xequil)

    # plot3d_anim(env.times[:-1], env.states[:-1], env.modes[:-1], env.ps_list, env.Nz_list, filename="gcas_gym.gif")


if __name__ == "__main__":
    # step_test()
    # trim_point_test()
    # neural_network_controller_test()
    GymEnvInterfaceTest()