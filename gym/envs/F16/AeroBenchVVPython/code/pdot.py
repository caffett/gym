'''
Stanley Bak
Python F-16
power derivative (pdot)
'''

from rtau import rtau, rtau_tf
import tensorflow as tf

def pdot(p3, p1):
    'pdot function'

    if p1 >= 50:
        if p3 >= 50:
            t = 5
            p2 = p1
        else:
            p2 = 60
            t = rtau(p2 - p3)
    else:
        if p3 >= 50:
            t = 5
            p2 = 40
        else:
            p2 = p1
            t = rtau(p2 - p3)

    pd = t * (p2 - p3)

    return pd

def pdot_tf(p3, p1):
    with tf.name_scope("pdot"):
        f1 = lambda: 5*(p1-p3)
        f2 = lambda: rtau_tf(60-p3)*(60-p3)
        f3 = lambda: 5*(40-p3)
        f4 = lambda: rtau_tf(p1-p3)*(p1-p3)
        func_dict = {
            tf.logical_and(tf.greater_equal(p1, 50), tf.greater_equal(p3, 50)): f1,
            tf.logical_and(tf.greater_equal(p1, 50), tf.less(p3, 50)): f2,
            tf.logical_and(tf.less(p1, 50), tf.greater_equal(p3, 50)): f3,
            tf.logical_and(tf.less(p1, 50), tf.less(p3, 50)): f4
        }

        ret = tf.case(func_dict, exclusive=True)

    return ret

def test_pdot_tf():
    p3 = 51
    p1 = 54
    p3_tf = tf.constant(51, dtype=tf.float32)
    p1_tf = tf.constant(54, dtype=tf.float32)
    with tf.Session() as sess:
        print(sess.run(pdot_tf(p3_tf, p1_tf)))
    print(pdot(p3, p1))

    p3 = 51
    p1 = 49
    p3_tf = tf.constant(51, dtype=tf.float32)
    p1_tf = tf.constant(49, dtype=tf.float32)
    with tf.Session() as sess:
        print(sess.run(pdot_tf(p3_tf, p1_tf)))
    print(pdot(p3, p1))

    p3 = 49
    p1 = 51
    p3_tf = tf.constant(49, dtype=tf.float32)
    p1_tf = tf.constant(51, dtype=tf.float32)
    with tf.Session() as sess:
        print(sess.run(pdot_tf(p3_tf, p1_tf)))
    print(pdot(p3, p1))

    p3 = 49
    p1 = 48
    p3_tf = tf.constant(49, dtype=tf.float32)
    p1_tf = tf.constant(48, dtype=tf.float32)
    with tf.Session() as sess:
        print(sess.run(pdot_tf(p3_tf, p1_tf)))
    print(pdot(p3, p1))

if __name__ == "__main__":
    test_pdot_tf()