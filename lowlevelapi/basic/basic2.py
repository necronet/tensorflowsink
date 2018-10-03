import tensorflow as tf
import numpy as np

# Will be set this value later for now is just a float
x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)

my_data = [
    [0, 1,],
    [2, 3,],
    [4, 5,],
    [6, 7,],
]

slices = tf.data.Dataset.from_tensor_slices(my_data)
next_item = slices.make_one_shot_iterator().get_next()

z = x + y

sess = tf.Session()
print(sess.run(z, feed_dict={x:10.1,y:0.9}))
print(sess.run(z, feed_dict={x: [1, 3], y: [2, 4]}))

while True:
  try:
    print(sess.run(next_item))
  except tf.errors.OutOfRangeError:
    break
