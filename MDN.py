import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import math

NSAMPLE = 10000
x_data = np.float32(np.random.uniform(-10.5, 10.5, (1, NSAMPLE))).T
r_data = np.float32(np.random.normal(size=(NSAMPLE,1)))
y_data = np.float32(np.sin(0.75*x_data)*7.0+x_data*0.5+r_data*1.0)

temp_data = x_data
x_data = y_data
y_data = temp_data


plt.figure(figsize=(8, 8))
plot_out = plt.plot(x_data,y_data,'y+',alpha=0.3)
plt.show()

x = tf.placeholder(dtype=tf.float32,shape=[None,1])
y = tf.placeholder(dtype=tf.float32,shape=[None,1])

HIDDEN = 20
W = tf.Variable(tf.random_normal([1,HIDDEN],stddev=1.0,dtype=tf.float32))
b = tf.Variable(tf.random_normal([1,HIDDEN],stddev=1.0,dtype=tf.float32))

W_out = tf.Variable(tf.random_normal([HIDDEN,1],stddev=1.0,dtype=tf.float32))
b_out = tf.Variable(tf.random_normal([1,1],stddev=1.0,dtype=tf.float32))

hidden_layer = tf.nn.tanh(tf.matmul(x,W)+b)
y_out = tf.matmul(hidden_layer,W_out) + b_out

lossfunc = tf.nn.l2_loss(y_out-y)

train_op = tf.train.RMSPropOptimizer(learning_rate=0.1,decay=0.8).minimize(lossfunc)

sess = tf.InteractiveSession()
init = tf.global_variables_initializer()
sess.run(init)

NEPOCH = 1000
for  i  in range(NEPOCH):
    sess.run(train_op,feed_dict={x:x_data,y:y_data})

x_test = np.float32(np.arange(-10.5,10.5,0.01)) # x_test.shape (210,)
x_test = x_test.reshape(x_test.size,1)  # x_test.shape (210,1)
y_test = sess.run(y_out,feed_dict={x:x_test})

plt.figure(figsize=(8,8))
plt.plot(x_data,y_data,'r+',x_test,y_test,'b+',alpha=0.3)
plt.savefig("failure.png",dpi=1000)
plt.show()
sess.close()


