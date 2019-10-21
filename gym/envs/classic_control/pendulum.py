import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from os import path


import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras import Model

import CoRec
ROOT = path.dirname(path.abspath(CoRec.__file__))+"/envs/gym_approx/"
from tensorflow.keras import backend as K


class PendulumEnv(gym.Env):
    metadata = {
        'render.modes' : ['human', 'rgb_array'],
        'video.frames_per_second' : 30
    }

    def __init__(self):
        self.max_speed=8
        self.max_torque=2.
        self.dt=.05
        self.original = np.array([0.0, 0.0])
        self.viewer = None

        high = np.array([1., 1., self.max_speed])
        self.action_space = spaces.Box(low=-self.max_torque, high=self.max_torque, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)
        init_high = np.array([np.pi, 1])
        self.initial_space = spaces.Box(low=-init_high, high=init_high, dtype=np.float32)

        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self,u):
        th, thdot = self.state # th := theta

        g = 10.
        m = 1.
        l = 1.
        dt = self.dt

        u = np.clip(u, -self.max_torque, self.max_torque)[0]
        self.last_u = u # for rendering
        costs = (angle_normalize(th)**2 + .1*thdot**2 + .001*(u**2))/1000

        newthdot = thdot + (-3*g/(2*l) * np.sin(th + np.pi) + 3./(m*l**2)*u) * dt
        newth = th + newthdot*dt
        newthdot = np.clip(newthdot, -self.max_speed, self.max_speed) #pylint: disable=E1111

        self.state = np.array([newth, newthdot])

        if self.check_safe(self.state):
            return self._get_obs(), -costs, False, {}
        return self._get_obs(), -1.0, True, {}

    def reset(self, x0=None):
        high = np.array([np.pi, 1])
        if x0 is None:
            self.state = self.np_random.uniform(low=-high, high=high)
        else:
            self.state = x0
        self.last_u = None
        return self._get_obs()

    def check_safe(self, state):
        # safe constriants
        # return (state > -0.6).all() and (state < 0.6)
        return state[1] > -999 and state[1] < 999

    def approximator(self, x0, step=200, algo='a2c'):
        model_name = "Pendulum-v0"
        if step == 200:
            approx = load_model(ROOT+model_name+"/"+algo+"_nn_approx_1e+06_approx200.model")
            # print(approx.summary())

        new_input = tf.keras.Input(tensor=tf.reshape(x0, (1,2)))
        new_output = approx(new_input)
        new_model = Model(new_input, new_output)

        sess = K.get_session()

        return tf.reshape(new_model.output, (2, )), sess

    def _get_obs(self):
        theta, thetadot = self.state
        return np.array([np.cos(theta), np.sin(theta), thetadot])

    def render(self, mode='human'):
        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(500,500)
            self.viewer.set_bounds(-2.2,2.2,-2.2,2.2)
            rod = rendering.make_capsule(1, .2)
            rod.set_color(.8, .3, .3)
            self.pole_transform = rendering.Transform()
            rod.add_attr(self.pole_transform)
            self.viewer.add_geom(rod)
            axle = rendering.make_circle(.05)
            axle.set_color(0,0,0)
            self.viewer.add_geom(axle)
            fname = path.join(path.dirname(__file__), "assets/clockwise.png")
            self.img = rendering.Image(fname, 1., 1.)
            self.imgtrans = rendering.Transform()
            self.img.add_attr(self.imgtrans)

        self.viewer.add_onetime(self.img)
        self.pole_transform.set_rotation(self.state[0] + np.pi/2)
        if self.last_u:
            self.imgtrans.scale = (-self.last_u/2, np.abs(self.last_u)/2)

        return self.viewer.render(return_rgb_array = mode=='rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

def angle_normalize(x):
    return (((x+np.pi) % (2*np.pi)) - np.pi)
