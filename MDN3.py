from keras import backend as K
import tensorflow as tf
from keras.engine.topology import Layer
from mpl_toolkits.mplot3d import axes3d,Axes3D  #<-- Note the capitalization
import matplotlib.pyplot as  plt
import numpy as np
import math

"""
----------------------------Define the mixture components
----------------------------Keras Version
"""
def get_mixture_coef_keras(output,numComponents=24,outputDim=1):
    out_pi = output[:,:numComponents]
    out_sigma = output[:,numComponents:2*numComponents]
    out_mu = output[:,2*numComponents:]
    out_mu = K.reshape(out_mu,[-1,numComponents,outputDim])
    out_mu = K.permute_dimensions(out_mu,[1,0,2])

    # use softmax to normalize pi into prob distribution
    max_pi = K.max(out_pi,axis=1,keepdims=True)
    out_pi = out_pi - max_pi
    out_pi = K.exp(out_pi)
    normalize_pi = 1 /K.sum(out_pi,axis=1,keepdims=True)
    out_pi = normalize_pi * out_pi
    # use exponential to make sure sigma is positive
    out_sigma = K.exp(out_sigma)
    return out_pi,out_sigma,out_mu


"""
----------------------------Tensorflow version
"""

def get_mixture_coef_tf(output,KMIX=24):
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

'''
def get_mixture_coef(output, KMIX=24, OUTPUTDIM=1):
  out_pi = tf.placeholder(dtype=tf.float32, shape=[None,KMIX], name="mixparam")
  out_sigma = tf.placeholder(dtype=tf.float32, shape=[None,KMIX], name="mixparam")
  out_mu = tf.placeholder(dtype=tf.float32, shape=[None,KMIX*OUTPUTDIM], name="mixparam")
  splits = tf.split(1, 2 + OUTPUTDIM, output)
  out_pi = splits[0]
  out_sigma = splits[1]
  out_mu = tf.stack(splits[2:], axis=2)
  out_mu = tf.transpose(out_mu, [1,0,2])
  # use softmax to normalize pi into prob distribution
  max_pi = tf.reduce_max(out_pi, 1, keep_dims=True)
  out_pi = tf.subtract(out_pi, max_pi)
  out_pi = tf.exp(out_pi)
  normalize_pi = tf.reciprocal(tf.reduce_sum(out_pi, 1, keep_dims=True))
  out_pi = tf.multiply(normalize_pi, out_pi)
  # use exponential to make sure sigma is positive
  out_sigma = tf.exp(out_sigma)
  return out_pi, out_sigma, out_mu
'''


def tf_normal_keras(y,mu,sigma):
    oneDivSqrtTwoPI = 1/math.sqrt(2*math.pi)
    result = y-mu
    result = K.permute_dimensions(result,[2,1,0])
    result = result*(1/sigma +1e-8)
    result =  -K.square(result)/2
    result = K.exp(result)*(1/(sigma+1e-8))*oneDivSqrtTwoPI
    result = K.prod(result,axis = [0]) # reuslt = K.prod(result,axis=0)
    return result
"""
--------------------------------Stage 1: return the probability of a gaussian distribution
"""
def tf_normal_tf(y,mu,sigma):  # return the probability of gaussian distribution
    oneDivSqrtTwoPI = 1/math.sqrt(2*math.pi)
    result = tf.subtract(y,mu)
    result = tf.multiply(result,tf.reciprocal(sigma))
    result = -tf.square(result)/2
    return tf.multiply(tf.exp(result),tf.reciprocal(sigma))*oneDivSqrtTwoPI

'''
def tf_normal(y, mu, sigma):
  oneDivSqrtTwoPI = 1 / math.sqrt(2*math.pi)
  result = tf.subtract(y, mu)
  result = tf.transpose(result, [2,1,0])
  result = tf.multiply(result,tf.reciprocal(sigma + 1e-8))
  result = -tf.square(result)/2
  result = tf.multiply(tf.exp(result),tf.reciprocal(sigma + 1e-8))*oneDivSqrtTwoPI
  result = tf.reduce_prod(result, reduction_indices=[0])
  return result
'''


"""
-------------------------------Stage 2: calculate the costfunction(y|x) = -log(sum_{k}^{K} Pi_{k} gaussian_distribution)
"""
def get_lossfunc_keras(out_pi,out_sigma,out_mu,y):
    result = tf_normal_keras(y,out_mu,out_sigma)
    result = result*out_pi
    result =K.sum(result,axis=1,keepdims=True)
    result = -K.log(result + 1e-8)
    return K.mean(result)

"""
-------------------------------Stage 2: calculate the costfunction(y|x) = -log(sum_{k=1}^{K} Pi_{k} gaussian_distribution)
"""
def get_lossfunc_tf(out_pi,out_sigma,out_mu,y):
    result = tf_normal_tf(y,out_mu,out_sigma)
    result = tf.multiply(result,out_pi)
    result = tf.reduce_sum(result,1,keep_dims=True)
    result = -tf.log(result)
    return tf.reduce_mean(result)


"""
-------------------------------Stage 2: calculate the costfunction(y|x) = -log(sum_{k=1}^{K} Pi_{k} gaussian_distribution)
-------------------------------add return kernel,before log
"""

def get_lossfunc(out_pi,out_sigma,out_mu,y):
    result = tf_normal_keras(y,out_mu,out_sigma)
    kernel = result
    result = tf.multiply(result,out_pi)
    result = tf.reduce_sum(result,1,keepdims=True)
    before_log = result
    result = -tf.log(result + 1e-8)
    return tf.reduce_sum(result),kernel,before_log

def get_pi_idx(x,pdf):
    N = pdf.size  # pdf.size = 24
    accumulate = 0
    for i in range(N):
        accumulate += pdf[i]
        if(accumulate >= x):
            return i
        else:
            print("Error with sampling ensemble")
            return -1

def generate_ensemble_tf(out_pi,out_mu,out_sigma,M = 10):
    x_test = np.float32(np.arange(-15.0, 15.0, 0.1))
    x_test = x_test.reshape(x_test.size, 1)

    NTEST = x_test.size # 300
    result = np.random.rand(NTEST,M) # initial random [0,1]
    rn = np.random.randn(NTEST,M)    # initial random [-1,1]
    mu = 0
    std = 0
    idx = 0
    # tranforms result into random ensembles
    for j in range(M):
        for i in range(NTEST):
            idx = get_pi_idx(result[i,j],out_pi[i])
            mu = out_mu[i,idx]
            std = out_sigma[i,idx]
            result[i,j] = mu + rn[i,j]*std
    return result


def generate_ensemble_keras(out_pi, out_mu, out_sigma, x_test, M = 10, OUTPUTDIM=1):
  NTEST = x_test.size
  result = np.random.rand(NTEST, M, OUTPUTDIM) # initially random [0, 1]
  rn = np.random.randn(NTEST, M) # normal random matrix (0.0, 1.0)
  mu = 0
  std = 0
  idx = 0
  # transforms result into random ensembles
  for j in range(0, M):
    for i in range(0, NTEST):
      for d in range(0, OUTPUTDIM):
        idx = np.random.choice(24, 1, p=out_pi[i])
        mu = out_mu[i, idx]
        std = out_sigma[i, idx]
        result[i, j, d] = mu + rn[i, j]*std
  return result

def mdn_loss(numComponents=24,outputDim=1):
    def loss(y,output):
        out_pi,out_sigma,out_mu = get_mixture_coef_keras(output,numComponents,outputDim)
        return get_lossfunc_keras(out_pi,out_sigma,out_mu,y)
    return loss


"""
-------------------------------Define a regular neural network: tensorflow.Version-------------------------------
"""

def regularnn(NHIDDEN =24,INPUTDIM=1,OUTPUTDIM=1,STDDEV=0.5):
    x = tf.placeholder(dtype=tf.float32,shape=[None,INPUTDIM], name = "x")
    y = tf.placeholder(dtype=tf.float32,shape=[None,OUTPUTDIM], name = "y")
    Wh = tf.Variable(tf.random_normal([INPUTDIM,NHIDDEN],stddev=STDDEV,dtype=tf.float32))
    bh = tf.Variable(tf.random_normal([NHIDDEN],stddev=STDDEV,dtype=tf.float32))
    W_out = tf.Variable(tf.random_normal([NHIDDEN,OUTPUTDIM],stddev=STDDEV,dtype=tf.float32))
    b_out = tf.Variable(tf.random_normal([OUTPUTDIM],stddev=STDDEV,dtype=tf.float32))
    hidden_layer = tf.nn.tanh(tf.matmul(x,Wh) + bh)
    output = tf.matmul(hidden_layer,W_out) + b_out
    return x,y,output

"""
-------------------------------Define a mixture density neural network: tensorflow.Version-------------------------------
"""

def mdn(NHIDDEN=24, INPUTDIM=1, OUTPUTDIM=1, STDEV=0.5, KMIX=24):
  NOUT = KMIX * (2+OUTPUTDIM)
  x = tf.placeholder(dtype=tf.float32, shape=[None,INPUTDIM], name="x")
  y = tf.placeholder(dtype=tf.float32, shape=[None,OUTPUTDIM], name="y")
  Wh = tf.Variable(tf.random_normal([INPUTDIM,NHIDDEN], stddev=STDEV, dtype=tf.float32))
  bh = tf.Variable(tf.random_normal([NHIDDEN], stddev=STDEV, dtype=tf.float32))
  Wo = tf.Variable(tf.random_normal([NHIDDEN,NOUT], stddev=STDEV, dtype=tf.float32))
  bo = tf.Variable(tf.random_normal([NOUT], stddev=STDEV, dtype=tf.float32))
  hidden_layer = tf.nn.tanh(tf.matmul(x, Wh) + bh)
  output = tf.matmul(hidden_layer,Wo) + bo
  return x,y,output


class MixtureDensity(Layer):
    def __init__(self, kernelDim, numComponents, **kwargs):
        self.hiddenDim = 24
        self.kernelDim = kernelDim
        self.numComponents = numComponents
        super(MixtureDensity, self).__init__(**kwargs)

    def build(self, inputShape):
        self.inputDim = inputShape[1]
        self.outputDim = self.numComponents * (2+self.kernelDim)
        self.Wh = K.variable(np.random.normal(scale=0.5,size=(self.inputDim, self.hiddenDim)))
        self.bh = K.variable(np.random.normal(scale=0.5,size=(self.hiddenDim)))
        self.Wo = K.variable(np.random.normal(scale=0.5,size=(self.hiddenDim, self.outputDim)))
        self.bo = K.variable(np.random.normal(scale=0.5,size=(self.outputDim)))

        self.trainable_weights = [self.Wh,self.bh,self.Wo,self.bo]

    def call(self, x, mask=None):
        hidden = K.tanh(K.dot(x, self.Wh) + self.bh)
        output = K.dot(hidden,self.Wo) + self.bo
        return output

    def get_output_shape_for(self, inputShape):
        return (inputShape[0], self.outputDim)


def oned2oned():
    NSAMPLE = 250

    y_data = np.float32(np.random.uniform(-10.5,10.5,(1,NSAMPLE))).T
    r_data = np.float32(np.random.normal(size = (NSAMPLE,1)))
    x_data = np.float32(np.sin(0.75*y_data)*7.0+y_data*0.5+r_data*1.0)

    x,y,output = mdn()
    out_pi,out_sigma,out_mu = get_mixture_coef_keras(output)
    lossfunc,k,bl = get_lossfunc(out_pi,out_sigma,out_mu,y)
    train_op = tf.train.AdamOptimizer().minimize(lossfunc)

    sess = tf.InteractiveSession()
    init = tf.global_variables_initializer()
    sess.run(init)

    plt.figure(figsize=(8,8))
    plt.plot(x_data,y_data,'ro',alpha=0.3)
    plt.show()

    NEPOCH = 10000
    loss = np.zeros(NEPOCH)
    for i in range(NEPOCH):
        sess.run(train_op,feed_dict={x:x_data,y:y_data})
        loss[i] = sess.run(lossfunc,feed_dict={x:x_data,y:y_data})
        print(loss[i])
    plt.figure(figsize=(8,8))
    plt.plot(np.arange(100,NEPOCH,1),loss[100:],'r-')
    plt.show()

    x_test = np.float32(np.arange(-15,15,0.1))
    NTEST = x_test.size
    x_test = x_test.reshape(NTEST,1)  # need to be a matrix, not a vector

    out_pi_test, out_sigma_test, out_mu_test = sess.run(get_mixture_coef_keras(output),feed_dict={x:x_test})

    y_test = generate_ensemble_keras(out_pi_test,out_sigma_test,out_mu_test,x_test,M=1)

    plt.figure(figsize=(8,8))
    plt.plot(x_data,y_data,'ro',x_test,y_test[:,:,0],'bo',alpha = 0.3)
    plt.show()

    # 1d to 2d test case
def oned2twod():
    NSAMPLE = 250
    fig = plt.figure()
    ax = Axes3D(fig)
    z_data = np.float32(np.random.uniform(-10.5,10.5,(1,NSAMPLE))).T
    r_data = np.float32(np.random.normal(size=(NSAMPLE,1)))
    x1_data = np.float32(np.sin(0.75*z_data)*7.0+z_data*0.5+r_data*1.0)
    x2_data = np.float32(np.sin(0.75*z_data)*7.0+z_data*0.5+r_data*1.0)

    ax.scatter(x1_data,x2_data)
    ax.legend()
    plt.show()

    x_data = np.dstack((x1_data,x2_data))
    x,y,output = mdn(INPUTDIM=1,OUTPUTDIM=2)
    out_pi,out_sigma,out_mu = get_mixture_coef_keras(output,outputDim=2)
    lossfunc,kernel,beforelog = get_lossfunc(out_pi,out_sigma,out_mu,y)

    train_op = tf.train.AdamOptimizer().minimize(lossfunc)

    sess = tf.InteractiveSession()
    init = tf.global_variables_initializer()
    sess.run(init)

    NEPOCH = 10000
    loss = np.zeros(NEPOCH)

    for i in range(NEPOCH):
        sess.run(train_op,feed_dict={x:z_data,y:x_data[:,0,:]})
        loss[i] = sess.run(lossfunc,feed_dict={x:z_data,y:x_data[:,0,:]})
        print(str(i)+":"+str(loss[i]))
        # loss[i],k,bl = sess.run([lossfunc,kernel,beforelog],feed_dict={x:z_data,y:x_data[:,0,:]})
        # print(str(i)+":"+str(loss[i])+","+str(k)+""+str(bl))

    plt.figure(figsize=(8,8))
    plt.plot(np.arange(100,NEPOCH,1),loss[100:],'r-')
    plt.show()


    x_test = np.float32(np.arange(-10.5,10.5,0.1))
    plt.plot(np.arange(-10.5,10.5,0.1))
    NTEST = x_test.size
    x_test = x_test.reshape(NTEST,1) # needs to be a matrix,not a vector

    out_pi_test,out_sigma_test,out_mu_test = sess.run(get_mixture_coef_keras(output,outputDim=2),feed_dict={x:x_test})

    y_test = generate_ensemble_keras(out_pi_test,out_mu_test,out_sigma_test,x_test,M=1,OUTPUTDIM=2)

    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(y_test[:,0,0],y_test[:,0,1],x_test,c='r')
    ax.scatter(x1_data,x2_data,z_data,c="b")
    ax.legend()
    plt.show()


oned2oned()
# oned2twod()
