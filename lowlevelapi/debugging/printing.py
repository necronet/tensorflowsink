import tensorflow as tf
x = tf.constant([[[1., 2.], [3., 4. ], [5. , 6. ]],
                 [[7., 8.], [9., 10.], [11., 12.]]])


# res = tf.slice(x, [0, 0, 0], [1, 3, 2]) => [[[1. 2.],[3. 4.],[5. 6.]]]

# All first elements
# res = tf.slice(x, [0, 0, 0], [2, 3, 1]) => [[[ 1.],[ 3.],[ 5.]], [[ 7.],[ 9.],[11.]]]

# All elements of the second matrix
# res = tf.slice(x, [1, 0, 0], [-1, -1, -1]) => [[[ 7,  8],[ 9, 10.],[11, 12]]]
# res = tf.slice(x, [1, 0, 0], [1, 3, 2])

# Getting last elements of the second matrix
# res = tf.slice(x, [0, 0, 0], [2, 3, 1]) => [[[ 1.],[ 3.],[ 5.]], [[ 7.],[ 9.],[11.]]]

t = tf.constant([[[1, 1, 1], [2, 2, 2]],
                 [[3, 3, 3], [4, 4, 4]],
                 [[5, 5, 5], [6, 6, 6]]])

res = tf.strided_slice(t, [0, 0, 0], [-1, 1, 2])

with tf.Session() as sess:
    sess.run(t)
    print(res.eval())
