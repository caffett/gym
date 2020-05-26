'''
Stanley Bak
Python F-16

Rtau function
'''

import tensorflow as tf

def rtau(dp):
    'rtau function'

    if dp <= 25:
        rt = 1.0
    elif dp >= 50:
        rt = .1
    else:
        rt = 1.9 - .036 * dp

    return rt

def rtau_tf(dp):
	with tf.name_scope("rtau"):
		f1 = lambda: tf.constant(1.0, dtype=tf.float32)
		f2 = lambda: tf.constant(0.1, dtype=tf.float32)
		f3 = lambda: 1.9 - .036 * dp
		rt = tf.case({tf.less_equal(dp, 25): f1, tf.greater_equal(dp, 50): f2}, default=f3, exclusive=True)
		return rt

def test_rtau_tf():
	dp = 24
	dp_tf = tf.constant(24, dtype=tf.float32)
	with tf.Session() as sess:
		print(sess.run(rtau_tf(dp_tf)))
	print(rtau(dp))

	dp = 26
	dp_tf = tf.constant(26, dtype=tf.float32)
	with tf.Session() as sess:
		print(sess.run(rtau_tf(dp_tf)))
	print(rtau(dp))

	dp = 51
	dp_tf = tf.constant(51, dtype=tf.float32)
	with tf.Session() as sess:
		print(sess.run(rtau_tf(dp_tf)))
	print(rtau(dp))

if __name__ == "__main__":

	test_rtau_tf()