'''
Stanley Bak
adc.py for F-16 model
'''

from math import sqrt

def adc(vt, alt):
    '''converts velocity (vt) and altitude (alt) to mach number (amach) and dynamic pressure (qbar)

    See pages 63-65 of Stevens & Lewis, "Aircraft Control and Simulation", 2nd edition
    '''

    # vt = freestream air speed

    ro = 2.377e-3
    tfac = 1 - .703e-5 * alt

    if alt >= 35000: # in stratosphere
        t = 390
    else:
        t = 519 * tfac # 3 rankine per atmosphere (3 rankine per 1000 ft)

    # rho = freestream mass density
    rho = ro * tfac**4.14

    # a = speed of sound at the ambient conditions
    # speed of sound in a fluid is the sqrt of the quotient of the modulus of elasticity over the mass density
    a = sqrt(1.4 * 1716.3 * t)

    # amach = mach number
    amach = vt / a

    # qbar = dynamic pressure
    qbar = .5 * rho * vt * vt

    return amach, qbar

import tensorflow as tf

def adc_tf(vt, alt):
    with tf.name_scope("adc"):
        ro = 2.377e-3
        tfac = 1 - .703e-5 * alt

        t = tf.cond(tf.greater_equal(alt, 35000), lambda: 390.0, lambda: 519 * tfac)
        rho = ro * tfac**4.14
        a = tf.sqrt(1.4 * 1716.3 * t)
        amach = vt / a
        qbar = .5 * rho * vt * vt

        return amach, qbar

def test_adc_tf():
    vt = 10
    alt = 34999
    print(adc(vt, alt))
    vt_tf = tf.constant(10, dtype=tf.float32)
    alt_tf = tf.constant(34999, dtype=tf.float32)
    with tf.Session() as sess:
        print(sess.run(adc_tf(vt_tf, alt_tf)))

    vt = 10
    alt = 35000
    print(adc(vt, alt))
    vt_tf = tf.constant(10, dtype=tf.float32)
    alt_tf = tf.constant(35000, dtype=tf.float32)
    with tf.Session() as sess:
        print(sess.run(adc_tf(vt_tf, alt_tf)))

if __name__ == "__main__":
    test_adc_tf()
