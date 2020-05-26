'''
Stanle Bak
Python F-16
Thrust function
'''

import numpy as np
import tensorflow as tf

from util import fix, fix_tf

def thrust(power, alt, rmach):
    'thrust lookup-table version'

    a = np.array([[1060, 670, 880, 1140, 1500, 1860], \
        [635, 425, 690, 1010, 1330, 1700], \
        [60, 25, 345, 755, 1130, 1525], \
        [-1020, -170, -300, 350, 910, 1360], \
        [-2700, -1900, -1300, -247, 600, 1100], \
        [-3600, -1400, -595, -342, -200, 700]], dtype=float).T

    b = np.array([[12680, 9150, 6200, 3950, 2450, 1400], \
        [12680, 9150, 6313, 4040, 2470, 1400], \
        [12610, 9312, 6610, 4290, 2600, 1560], \
        [12640, 9839, 7090, 4660, 2840, 1660], \
        [12390, 10176, 7750, 5320, 3250, 1930], \
        [11680, 9848, 8050, 6100, 3800, 2310]], dtype=float).T

    c = np.array([[20000, 15000, 10800, 7000, 4000, 2500], \
        [21420, 15700, 11225, 7323, 4435, 2600], \
        [22700, 16860, 12250, 8154, 5000, 2835], \
        [24240, 18910, 13760, 9285, 5700, 3215], \
        [26070, 21075, 15975, 11115, 6860, 3950], \
        [28886, 23319, 18300, 13484, 8642, 5057]], dtype=float).T

    if alt < 0:
        alt = 0.01 # uh, why not 0?

    h = .0001 * alt

    i = fix(h)

    if i >= 5:
        i = 4

    dh = h - i
    rm = 5 * rmach
    m = fix(rm)

    if m >= 5:
        m = 4
    elif m <= 0:
        m = 0

    dm = rm - m
    cdh = 1 - dh

    # do not increment these, since python is 0-indexed while matlab is 1-indexed
    #i = i + 1
    #m = m + 1

    s = b[i, m] * cdh + b[i + 1, m] * dh
    t = b[i, m + 1] * cdh + b[i + 1, m + 1] * dh
    tmil = s + (t - s) * dm

    if power < 50:
        s = a[i, m] * cdh + a[i + 1, m] * dh
        t = a[i, m + 1] * cdh + a[i + 1, m + 1] * dh
        tidl = s + (t - s) * dm
        thrst = tidl + (tmil - tidl) * power * .02
    else:
        s = c[i, m] * cdh + c[i + 1, m] * dh
        t = c[i, m + 1] * cdh + c[i + 1, m + 1] * dh
        tmax = s + (t - s) * dm
        thrst = tmil + (tmax - tmil) * (power - 50) * .02

    return thrst


def thrust_tf(power, alt, rmach):
    with tf.name_scope("threst"):
        a = tf.constant(np.array([[1060, 670, 880, 1140, 1500, 1860], \
            [635, 425, 690, 1010, 1330, 1700], \
            [60, 25, 345, 755, 1130, 1525], \
            [-1020, -170, -300, 350, 910, 1360], \
            [-2700, -1900, -1300, -247, 600, 1100], \
            [-3600, -1400, -595, -342, -200, 700]], dtype=np.float32).T)

        b = tf.constant(np.array([[12680, 9150, 6200, 3950, 2450, 1400], \
            [12680, 9150, 6313, 4040, 2470, 1400], \
            [12610, 9312, 6610, 4290, 2600, 1560], \
            [12640, 9839, 7090, 4660, 2840, 1660], \
            [12390, 10176, 7750, 5320, 3250, 1930], \
            [11680, 9848, 8050, 6100, 3800, 2310]], dtype=np.float32).T)

        c = tf.constant(np.array([[20000, 15000, 10800, 7000, 4000, 2500], \
            [21420, 15700, 11225, 7323, 4435, 2600], \
            [22700, 16860, 12250, 8154, 5000, 2835], \
            [24240, 18910, 13760, 9285, 5700, 3215], \
            [26070, 21075, 15975, 11115, 6860, 3950], \
            [28886, 23319, 18300, 13484, 8642, 5057]], dtype=np.float32).T)
    
    with tf.name_scope("threst"):
        alt = tf.cond(tf.less(alt, 0), lambda: 0.01, lambda: alt)
        h = .0001 * alt
        i = fix_tf(h)
        i = tf.cond(tf.greater_equal(i, 5.0), lambda: 4.0, lambda: i)

        dh = h - i
        rm = 5 * rmach
        m = fix_tf(rm)
        m = tf.clip_by_value(m, 0, 4)

        dm = rm - m
        cdh = 1 - dh

        # do not increment these, since python is 0-indexed while matlab is 1-indexed
        #i = i + 1
        #m = m + 1
        i = tf.cast(i, tf.int32)
        m = tf.cast(m, tf.int32)
        s = b[i, m] * cdh + b[i + 1, m] * dh
        t = b[i, m + 1] * cdh + b[i + 1, m + 1] * dh
        tmil = s + (t - s) * dm

        def f1():
            s = a[i, m] * cdh + a[i + 1, m] * dh
            t = a[i, m + 1] * cdh + a[i + 1, m + 1] * dh
            tidl = s + (t - s) * dm
            thrst = tidl + (tmil - tidl) * power * .02

            return thrst

        def f2():
            s = c[i, m] * cdh + c[i + 1, m] * dh
            t = c[i, m + 1] * cdh + c[i + 1, m + 1] * dh
            tmax = s + (t - s) * dm
            thrst = tmil + (tmax - tmil) * (power - 50) * .02

            return thrst

        thrst = tf.cond(tf.less(power, 50), f1, f2)

    return thrst

def test_thrust_tf():
    def template(power, alt, rmach):
        power_tf = tf.constant(power, dtype=tf.float32)
        alt_tf = tf.constant(alt, dtype=tf.float32)
        rmach_tf = tf.constant(rmach, dtype=tf.float32)
        with tf.Session() as sess:
            print(sess.run(thrust_tf(power_tf, alt_tf, rmach_tf)))

        print(thrust(power, alt, rmach))

    # alt < 0, alt < 500, alt > 500
    # rmach < 0.8, rmach > 0.8
    # power < 50, power > 50

    for a in (-1, 499, 501):
        for r in (0.79, 0.81):
            for p in (49, 51):
                template(p, a, r)

if __name__ == "__main__":
    test_thrust_tf()