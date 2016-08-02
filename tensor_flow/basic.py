import tensorflow as tf

# # 1x2 matrix
# matrix1 = tf.constant([[3., 3.]])

# # 2x1 matrix
# matrix2 = tf.constant([[2.], [2.]])

# # multiplication
# product = tf.matmul(matrix1, matrix2)

# with tf.Session() as sess:
#   result = sess.run([product])
#   print(product)

# Create a Variable, that will be initialized to the scalar value 0.
state = tf.Variable(0, name="counter")

# Create an Op to add one to `state`.

one = tf.constant(1)
new_value = tf.add(state, one)
update = tf.assign(state, new_value)

# Variables must be initialized by running an `init` Op after having
# launched the graph.  We first have to add the `init` Op to the graph.
init_op = tf.initialize_all_variables()

# Launch the graph and run the ops.
with tf.Session() as sess:
  # Run the 'init' op
  sess.run(init_op)
  # Print the initial value of 'state'
  print(sess.run(state))
  # Run the op that updates 'state' and print 'state'.
  for _ in range(3):
    sess.run(update)
    print(sess.run(state))