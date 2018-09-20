import tensorflow as tf
import numpy as  np
import matplotlib.pyplot as plt
import math


NHIDDEN = 24
STDEV = 0.5
KMIX = 24 # number of mixtures
NOUT = KMIX*3 #pi,mu,stdev
oneDivSqrtTwoPI = 1/math.sqrt(2*math.pi) # normalization factor for gaussian,not needed

'''-
-----------------------Build Model-----------------------------------------------
'''
x = tf.placeholder(dtype=tf.float32,shape=[None,1],name="x")
y = tf.placeholder(dtype=tf.float32,shape=[None,1],name="y")

Wh = tf.Variable(tf.random_normal([1,NHIDDEN],stddev=STDEV,dtype=tf.float32))
bh = tf.Variable(tf.random_normal([1,NHIDDEN],stddev=STDEV,dtype=tf.float32))

Wo = tf.Variable(tf.random_normal([NHIDDEN,NOUT],stddev=STDEV,dtype=tf.float32))
bo = tf.Variable(tf.random_normal([1,NOUT],stddev=STDEV,dtype=tf.float32))

hidden_layer = tf.nn.tanh(tf.matmul(x,Wh) +bh)
output = tf.matmul(hidden_layer,Wo) + bo


"""

-------------------------------Define mixture function----------------------------------

"""

def get_mixture_coef(output):
    out_pi = tf.placeholder(dtype=tf.float32,shape=[None,KMIX],name="mixparam")
    out_sigma = tf.placeholder(dtype=tf.float32,shape=[None,KMIX],name="mixparam")
    out_mu = tf.placeholder(dtype=tf.float32,shape=[None,KMIX],name="mixparam")

    out_pi,out_sigma,out_mu = tf.split(output,3,1)  # split tensor[None,KMIX(72)] into [NONE,24] [NONE,24] [NONE,24]
    max_pi = tf.reduce_max(out_pi,1,keep_dims=True)
    out_pi = tf.subtract(out_pi,max_pi)

    out_pi = tf.exp(out_pi)

    normalize_pi = tf.reciprocal(tf.reduce_sum(out_pi,1,keep_dims=True))
    out_pi = tf.multiply(normalize_pi,out_pi)
    out_sigma = tf.exp(out_sigma)

    return out_pi,out_sigma,out_mu
"""
--------------------------------define loss function--------------------------------------
"""
"""
--------------------------------Stage 1: return the probability of a gaussian distribution
"""
def tf_normal(x,mu,sigma):  # return the probability of gaussian distribution
    result = tf.subtract(x,mu)
    result = tf.multiply(result,tf.reciprocal(sigma))
    result = -tf.square(result)/2
    return tf.multiply(tf.exp(result),tf.reciprocal(sigma))*oneDivSqrtTwoPI

"""
-------------------------------Stage 2: calculate the costfunction(y|x) = -log(sum_{k}^{K} Pi_{k} gaussian_distribution)
"""
def get_lossfunc(out_pi,out_sigma,out_mu,y):
    result = tf_normal(y,out_mu,out_sigma)
    result = tf.multiply(result,out_pi)
    result = tf.reduce_sum(result,1,keep_dims=True)
    result = -tf.log(result)

    return tf.reduce_mean(result)

"""
-------------------------------Stage 3: DEFINE a  optimizer
"""




"""
To sample a mixed gaussian distribution,we randomly select which distribution based on the set of  PI_{k} probilities,
and then proceed to draw the point based of the Kth  gaussian distribution
"""
x_test = np.float32(np.arange(-15,15,0.1))
NTEST =x_test.size
x_test = x_test.reshape(NTEST,1)

def get_pi_idx(x,pdf):
    N = pdf.size  # pdf.size = 24
    accumulate = 0
    for i in range(N):
        accumulate += pdf[i]
        if(accumulate >=  x):
            return i
        else:
            print("Error with sampling ensemble")
            return -1

def generate_ensemble(out_pi,out_mu,out_sigma,M = 10):
    NTEST = x_test.size # 300
    result = np.random.rand(NTEST,M) # initial random [0,1]
    rn = np.random.randn(NTEST,M)
    mu = 0
    std = 0
    idx = 0
    # tranforms result into random ensembles
    for j in range(M):
        for i in range(NTEST):
            idx = get_pi_idx(result[i,j],out_pi[i])
            mu = out_mu[i,idx]
            result[i,j] = mu + rn[i,j]*std
        return result

""""
Let's see how the generated data looks like:Experiment start
"""


NEPOCH = 10000
NSAMPLE = 2500
y_data = np.float32(np.random.uniform(-10.5, 10.5, (1, NSAMPLE))).T
r_data = np.float32(np.random.normal(size=(NSAMPLE, 1)))  # random noise
x_data = np.float32(np.sin(0.75 * y_data) * 7.0 + y_data * 0.5 + r_data * 1.0)

out_pi,out_sigma,out_mu = get_mixture_coef(output)
lossfunc =  get_lossfunc(out_pi,out_sigma,out_mu,y)
train_op = tf.train.AdamOptimizer().minimize(lossfunc)

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    loss = np.zeros(NEPOCH)
    for i in range(NEPOCH):
        sess.run(train_op,feed_dict={x:x_data,y:y_data})
        loss[i] = sess.run(lossfunc,feed_dict={x:x_data,y:y_data})

    out_pi_test,out_sigma_test,out_mu_test =sess.run(get_mixture_coef(output),feed_dict={x:x_test})
    y_test = generate_ensemble(out_pi_test,out_mu_test,out_sigma_test)

    plt.figure(figsize=(8,8))
    plt.plot(x_data,y_data,'ro',x_test,y_test,'bo',alpha = 0.3)
    plt.show()

















