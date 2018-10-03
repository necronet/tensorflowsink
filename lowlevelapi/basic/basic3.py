import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

# kinit = np.ones((1,1)) * 0.2
# kernel_initializer = tf.constant_initializer(kinit)
# kernel_initializer=kernel_initializer

x = tf.placeholder(tf.float32, shape=[1,1])
linear_model = tf.layers.Dense(units=1, use_bias=False)
y = linear_model(x)

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)
output = sess.run(y, {x:[[1]]})
w = linear_model.get_weights()

print('{}*[[1]] = {}'.format(w,output))



#writer = tf.summary.FileWriter('.')
#writer.add_graph(tf.get_default_graph())
