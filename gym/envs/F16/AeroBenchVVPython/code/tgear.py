'''
Stanley Bak
Python F-16 GCAS
'''
import tensorflow as tf

def tgear(thtl):
    'tgear function'

    if thtl <= .77:
        tg = 64.94 * thtl
    else:
        tg = 217.38 * thtl - 117.38

    return tg

def tgear_tf(thtl):
	with tf.name_scope("tgear"):
		thtl077_tf = tf.constant(0.77, dtype=tf.float32)
		return tf.cond(tf.less_equal(thtl, thtl077_tf), lambda: 64.94 * thtl, lambda: 217.38 * thtl - 117.38)

def test_tgear_tf():
	thtl = 0.76
	thtl_tf = tf.constant(0.76, dtype=tf.float32)
	print(tgear(thtl))
	with tf.Session() as sess:
		print(sess.run(tgear_tf(thtl_tf)))

	thtl = 0.78
	thtl_tf = tf.constant(0.78, dtype=tf.float32)
	print(tgear(thtl))
	with tf.Session() as sess:
		print(sess.run(tgear_tf(thtl_tf)))

if __name__ == "__main__":
	test_tgear_tf()