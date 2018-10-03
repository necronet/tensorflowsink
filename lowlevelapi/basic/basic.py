import tensorflow as tf
import numpy as np


# Tensorflow programs consist in two elements:

# Graph Building

a = tf.constant(3.0, dtype=tf.float32)
b = tf.constant(4.0)
total = a + b
print(a)
print(b)
print(total)

vec = tf.random_uniform(shape=(3,))
out1 = vec + 1
out2 = vec + 2

# TensorBoard (feature)
writer = tf.summary.FileWriter('.')
writer.add_graph(tf.get_default_graph())

# Graph running
sess = tf.Session()

#print(sess.run(total))

# Passing dictionary to Session
# print(sess.run({'ab':(a, b), 'total':total}))

print(sess.run(vec))
print(sess.run(vec))
print(sess.run((out1, out2)))
