import tensorflow as tf

constant = tf.constant([5,3.0,4.5], dtype=tf.float32,name='Constant_init')
print (constant)

placeholder = tf.placeholder(tf.float32, name='Placeholder_init')
print(placeholder)

variable = tf.Variable(3, dtype=tf.float32, name='Variable_init')
print(variable)

matrix = tf.zeros([3,4], tf.int32, name='Matrix_zeros')
print(matrix)


mult = placeholder*constant
result =  session.run(mult,feed_dict={placeholder:[[15,10,3]]})
print(result)

A = tf.placeholder(tf.float32, shape=(2,2))
B = tf.placeholder(tf.float32, shape=(2,3))

'''
[1,0]
[0,0]

[1,0]
[0,0]
[1,7]
'''
mult = tf.matmul(A,B)
init = tf.global_variables_initializer()
session = tf.Session()
session.run(init)
print(session.run(mult, feed_dict={A:[[1,2],[2,2]], B:[[12,3,4],[5,7,3]]}))

C = tf.placeholder(tf.float32, shape=(2))
D = tf.placeholder(tf.float32, shape=(2))
dot = tf.tensordot(C,D,1)
session = tf.Session()
session.run(init)
print(session.run(dot, feed_dict={C:[1,2],D:[3,4]}))

