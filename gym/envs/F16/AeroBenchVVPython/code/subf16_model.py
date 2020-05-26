'''
Stanley Bak
Python F-16 subf16
outputs aircraft state vector deriative
'''

#         x[0] = air speed, VT    (ft/sec)
#         x[1] = angle of attack, alpha  (rad)
#         x[2] = angle of sideslip, beta (rad)
#         x[3] = roll angle, phi  (rad)
#         x[4] = pitch angle, theta  (rad)
#         x[5] = yaw angle, psi  (rad)
#         x[6] = roll rate, P  (rad/sec)
#         x[7] = pitch rate, Q  (rad/sec)
#         x[8] = yaw rate, R  (rad/sec)
#         x[9] = northward horizontal displacement, pn  (feet)
#         x[10] = eastward horizontal displacement, pe  (feet)
#         x[11] = altitude, h  (feet)
#         x[12] = engine thrust dynamics lag state, pow
#
#         u[0] = throttle command  0.0 < u(1) < 1.0
#         u[1] = elevator command in degrees
#         u[2] = aileron command in degrees
#         u[3] = rudder command in degrees
#

from math import sin, cos, pi

from adc import adc, adc_tf
from tgear import tgear, tgear_tf
from pdot import pdot, pdot_tf
from thrust import thrust, thrust_tf
from cx import cx
from cy import cy
from cz import cz
from cl import cl
from dlda import dlda
from dldr import dldr
from cm import cm
from cn import cn
from dnda import dnda
from dndr import dndr
from dampp import dampp, dampp_tf

from Morellif16 import Morellif16

def subf16_model(x, u, model, adjust_cy=True, multipliers=None):
    '''output aircraft state vector derivative for a given input

    The reference for the model is Appendix A of Stevens & Lewis

    if multipliers is not None, it should be a 7-tuple:
    xcg_mult, cxt_mult, cyt_mult, czt_mult, clt_mult, cmt_mult, cnt_mult

    xcg is x component of center of gravity (between 0.0 and 1.0, default 0.35)

    cxt is the x-axis aerodynamic force coefficient
    cyt is the sideforce coefficient
    czt is the z-axis force coefficient

    clt is a sum of the rolling moment coefficients
    cmt is the pitching moment coefficient
    cnt is a sum of the yawing moment coefficients
    '''

    assert model == 'stevens' or model == 'morelli'
    assert len(x) == 13
    assert len(u) == 4
    assert multipliers is None or len(multipliers) == 7

    xcg = 0.35

    if multipliers is not None:
        xcg *= multipliers[0]

    thtlc, el, ail, rdr = u

    s = 300
    b = 30
    cbar = 11.32
    rm = 1.57e-3
    xcgr = .35
    he = 160.0
    c1 = -.770
    c2 = .02755
    c3 = 1.055e-4
    c4 = 1.642e-6
    c5 = .9604
    c6 = 1.759e-2
    c7 = 1.792e-5
    c8 = -.7336
    c9 = 1.587e-5
    rtod = 57.29578
    g = 32.17

    xd = x.copy()
    vt = x[0]
    alpha = x[1]*rtod
    beta = x[2]*rtod
    phi = x[3]
    theta = x[4]
    psi = x[5]
    p = x[6]
    q = x[7]
    r = x[8]
    alt = x[11]
    power = x[12]

    # air data computer and engine model
    amach, qbar = adc(vt, alt)
    cpow = tgear(thtlc)

    xd[12] = pdot(power, cpow)

    t = thrust(power, alt, amach)
    dail = ail/20
    drdr = rdr/30

    # component build up

    if model == 'stevens':
        # stevens & lewis (look up table version)
        cxt = cx(alpha, el)
        cyt = cy(beta, ail, rdr)
        czt = cz(alpha, beta, el)

        clt = cl(alpha, beta) + dlda(alpha, beta) * dail + dldr(alpha, beta) * drdr
        cmt = cm(alpha, el)
        cnt = cn(alpha, beta) + dnda(alpha, beta) * dail + dndr(alpha, beta) * drdr
    else:
        # morelli model (polynomial version)
        cxt, cyt, czt, clt, cmt, cnt = Morellif16(alpha*pi/180, beta*pi/180, el*pi/180, ail*pi/180, rdr*pi/180, \
                                                  p, q, r, cbar, b, vt, xcg, xcgr)

    # multipliers adjustement
    if multipliers is not None:
        cxt *= multipliers[1]
        cyt *= multipliers[2]
        czt *= multipliers[3]

        clt *= multipliers[4]
        cmt *= multipliers[5]
        cnt *= multipliers[6]

    # add damping derivatives

    tvt = .5 / vt
    b2v = b * tvt
    cq = cbar * q * tvt

    # get ready for state equations
    d = dampp(alpha)
    cxt = cxt + cq * d[0]
    cyt = cyt + b2v * (d[1] * r + d[2] * p)
    czt = czt + cq * d[3]
    clt = clt + b2v * (d[4] * r + d[5] * p)
    cmt = cmt + cq * d[6] + czt * (xcgr-xcg)
    cnt = cnt + b2v * (d[7] * r + d[8] * p)-cyt * (xcgr-xcg) * cbar/b
    cbta = cos(x[2])
    u = vt * cos(x[1]) * cbta
    v = vt * sin(x[2])
    w = vt * sin(x[1]) * cbta
    sth = sin(theta)
    cth = cos(theta)
    sph = sin(phi)
    cph = cos(phi)
    spsi = sin(psi)
    cpsi = cos(psi)
    qs = qbar * s
    qsb = qs * b
    rmqs = rm * qs
    gcth = g * cth
    qsph = q * sph
    ay = rmqs * cyt
    az = rmqs * czt

    # force equations
    udot = r * v-q * w-g * sth + rm * (qs * cxt + t)
    vdot = p * w-r * u + gcth * sph + ay
    wdot = q * u-p * v + gcth * cph + az
    dum = (u * u + w * w)

    xd[0] = (u * udot + v * vdot + w * wdot)/vt
    xd[1] = (u * wdot-w * udot)/dum
    xd[2] = (vt * vdot-v * xd[0]) * cbta/dum

    # kinematics
    xd[3] = p + (sth/cth) * (qsph + r * cph)
    xd[4] = q * cph-r * sph
    xd[5] = (qsph + r * cph)/cth

    # moments
    xd[6] = (c2 * p + c1 * r + c4 * he) * q + qsb * (c3 * clt + c4 * cnt)

    xd[7] = (c5 * p-c7 * he) * r + c6 * (r * r-p * p) + qs * cbar * c7 * cmt
    xd[8] = (c8 * p-c2 * r + c9 * he) * q + qsb * (c4 * clt + c9 * cnt)

    # navigation
    t1 = sph * cpsi
    t2 = cph * sth
    t3 = sph * spsi
    s1 = cth * cpsi
    s2 = cth * spsi
    s3 = t1 * sth-cph * spsi
    s4 = t3 * sth + cph * cpsi
    s5 = sph * cth
    s6 = t2 * cpsi + t3
    s7 = t2 * spsi-t1
    s8 = cph * cth
    xd[9] = u * s1 + v * s3 + w * s6 # north speed
    xd[10] = u * s2 + v * s4 + w * s7 # east speed
    xd[11] = u * sth-v * s5-w * s8 # vertical speed

    # outputs

    xa = 15.0                  # sets distance normal accel is in front of the c.g. (xa = 15.0 at pilot)
    az = az-xa * xd[7]           # moves normal accel in front of c.g.

    ####################################
    ###### peter additionls below ######
    if adjust_cy:
        ay = ay+xa*xd[8]           # moves side accel in front of c.g.

    # For extraction of Nz
    Nz = (-az / g) - 1 # zeroed at 1 g, positive g = pulling up
    Ny = ay / g

    return xd, Nz, Ny, az, ay


import tensorflow as tf
def subf16_model_tf(x, u, multipliers):
    with tf.name_scope("subf16_model"):
        xcg = 0.35
        if multipliers is not None:
            xcg *= multipliers[0]

        thtlc = u[0]
        el = u[1]
        ail = u[2]
        rdr = u[3]

        s = 300
        b = 30
        cbar = 11.32
        rm = 1.57e-3
        xcgr = .35
        he = 160.0
        c1 = -.770
        c2 = .02755
        c3 = 1.055e-4
        c4 = 1.642e-6
        c5 = .9604
        c6 = 1.759e-2
        c7 = 1.792e-5
        c8 = -.7336
        c9 = 1.587e-5
        rtod = 57.29578
        g = 32.17

        vt = x[0]
        alpha = x[1]*rtod
        beta = x[2]*rtod
        phi = x[3]
        theta = x[4]
        psi = x[5]
        p = x[6]
        q = x[7]
        r = x[8]
        alt = x[11]
        power = x[12]

        # air data computer and engine model
        amach, qbar = adc_tf(vt, alt)
        cpow = tgear_tf(thtlc)

        xd12 = pdot_tf(power, cpow)

        t = thrust_tf(power, alt, amach)

        # morelli model (polynomial version)
        cxt, cyt, czt, clt, cmt, cnt = Morellif16(alpha*pi/180, beta*pi/180, el*pi/180, ail*pi/180, rdr*pi/180, \
                                                      p, q, r, cbar, b, vt, xcg, xcgr)

        # multipliers adjustement
        if multipliers is not None:
            cxt *= multipliers[1]
            cyt *= multipliers[2]
            czt *= multipliers[3]

            clt *= multipliers[4]
            cmt *= multipliers[5]
            cnt *= multipliers[6]

        # add damping derivatives

        tvt = .5 / vt
        b2v = b * tvt
        cq = cbar * q * tvt

        # get ready for state equations
        d = dampp_tf(alpha)
        # input d[...] c*t r p
        # output c*t
        cxt = cxt + cq * d[0]
        cyt = cyt + b2v * (d[1] * r + d[2] * p)
        czt = czt + cq * d[3]
        clt = clt + b2v * (d[4] * r + d[5] * p)
        cmt = cmt + cq * d[6] + czt * (xcgr-xcg)
        cnt = cnt + b2v * (d[7] * r + d[8] * p)-cyt * (xcgr-xcg) * cbar/b

        cbta = tf.cos(x[2])
        u = vt * tf.cos(x[1]) * cbta
        v = vt * tf.sin(x[2])
        w = vt * tf.sin(x[1]) * cbta
        sth = tf.sin(theta)
        cth = tf.cos(theta)
        sph = tf.sin(phi)
        cph = tf.cos(phi)
        spsi = tf.sin(psi)
        cpsi = tf.cos(psi)

        qs = qbar * s
        qsb = qs * b
        rmqs = rm * qs
        gcth = g * cth
        qsph = q * sph
        ay = rmqs * cyt
        az = rmqs * czt

        # force equations
        udot = r * v-q * w-g * sth + rm * (qs * cxt + t)
        vdot = p * w-r * u + gcth * sph + ay
        wdot = q * u-p * v + gcth * cph + az
        dum = (u * u + w * w)

        xd0 = (u * udot + v * vdot + w * wdot)/vt
        xd1 = (u * wdot-w * udot)/dum
        xd2 = (vt * vdot-v * xd0) * cbta/dum

        # kinematics
        xd3 = p + (sth/cth) * (qsph + r * cph)
        xd4 = q * cph-r * sph
        xd5 = (qsph + r * cph)/cth

        # moments
        xd6 = (c2 * p + c1 * r + c4 * he) * q + qsb * (c3 * clt + c4 * cnt)

        xd7 = (c5 * p-c7 * he) * r + c6 * (r * r-p * p) + qs * cbar * c7 * cmt
        xd8 = (c8 * p-c2 * r + c9 * he) * q + qsb * (c4 * clt + c9 * cnt)

        # navigation
        t1 = sph * cpsi
        t2 = cph * sth
        t3 = sph * spsi

        s1 = cth * cpsi
        s2 = cth * spsi
        s3 = t1 * sth-cph * spsi
        s4 = t3 * sth + cph * cpsi
        s5 = sph * cth
        s6 = t2 * cpsi + t3
        s7 = t2 * spsi-t1
        s8 = cph * cth
        
        xd9 = u * s1 + v * s3 + w * s6 # north speed
        xd10 = u * s2 + v * s4 + w * s7 # east speed
        xd11 = u * sth-v * s5-w * s8 # vertical speed

        xd = tf.stack([xd0, xd1, xd2, xd3, xd4, xd5, xd6, xd7, xd8, xd9, xd10, xd11, xd12])

        # outputs
        xa = 15.0                  # sets distance normal accel is in front of the c.g. (xa = 15.0 at pilot)
        az = az-xa * xd7           # moves normal accel in front of c.g.
        ay = ay+xa*xd8           # moves side accel in front of c.g.

        # For extraction of Nz
        Nz = (-az / g) - 1 # zeroed at 1 g, positive g = pulling up
        Ny = ay / g

    return xd, Nz, Ny, az, ay#, xd12, power, cpow

import gym
import CoRec
import numpy as np

def test_subf16_model():
    env = gym.make("F16GCAS-v0")
    def template(x, u):
        x_tf = tf.constant(x, dtype=tf.float32)
        u_tf = tf.constant(u, dtype=tf.float32)

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

        with tf.Session() as sess:
            tf_ret = sess.run(subf16_model_tf(x_tf, u_tf, multipliers_tf))
            print(tf_ret)

        ret = subf16_model(x, u, "morelli", multipliers)
        print(ret)

        # diff = np.sum(np.sum(np.abs(np.array(tf_ret)-np.array(ret))))
        # print(diff)
        # if diff > 0.01:
        #     print("x:", x)
        #     print("u:", u)
        # print("------")

    TEST_ROUND = 1
    for _ in range(TEST_ROUND):
        #x = env.env.initial_space.sample()[:13]
        # x = env.env.stable_state[:13]
        x = np.array([ 5.4000000e+02,  3.7027162e-02,  0.0000000e+00,  7.8539819e-01,
                   -1.2566371e+00, -7.8539819e-01,  0.0000000e+00,  0.0000000e+00,
                    0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  3.6000000e+03,
                    9.0000000e+00])
        u = np.zeros((4,), dtype=np.float32)
        u0 = np.random.random()
        u123 = env.action_space.sample()
        u[0] = u0
        u[:3] = u123

        template(x, u)

if __name__ == "__main__":
    test_subf16_model()
