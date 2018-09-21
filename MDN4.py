import numpy as np
from scipy.stats import *
from scipy.stats import norm as normal
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
import keras.backend as K
from keras.layers import Dense

from sklearn.model_selection import train_test_split

np.random.seed(1)

def build_toy_dataset(nsample=40000):
    y_data = np.float32(np.random.uniform(-10.5, 10.5, (1, nsample))).T
    r_data = np.float32(np.random.normal(size=(nsample,1))) # random noise
    x_data = np.float32(np.sin(0.75*y_data)*7.0+y_data*0.5+r_data*1.0)
    return train_test_split(x_data, y_data, random_state=42, train_size=0.1)

X_train, X_test, y_train, y_test = build_toy_dataset()
print("Size of features in training data: {}".format(X_train.shape))
print("Size of output in training data: {}".format(y_train.shape))
print("Size of features in test data: {}".format(X_test.shape))
print("Size of output in test data: {}".format(y_test.shape))

# sns.regplot(X_train, y_train, fit_reg=False)

class MixtureDensityNetWork():
    """
       Mixture density network for outputs y on inputs x.
       p((x,y), (z,theta))
       = sum_{k=1}^K pi_k(x; theta) Normal(y; mu_k(x; theta), sigma_k(x; theta))
       where pi, mu, sigma are the output of a neural network taking x
       as input and with parameters theta. There are no latent variables
       z, which are hidden variables we aim to be Bayesian about.
       """
    def __init__(self,K):
        self.K = K # Here K is the amount of Mixtures
    def mapping(self,X):
        """
        :param X: pi,mu,sigma=NN(x;theta)
        :return:
        """
        hidden1 = Dense(15,activation='relu')(X)
        hidden2 = Dense(15,activation='relu')(hidden1)
        self.mus = Dense(self.K,activation='relu')(hidden2)
        self.sigmas = Dense(self.K,activation=K.exp)(hidden2)
        self.pi = Dense(self.K,activation=K.softmax)(hidden2)

    def log_prob(self,xs,zs=None):
        """
        log p((xs,ys),(z,theta))=sum_{n=1}^{N} log p((xs[n:],ys[n]),theta)
        :param xs:
        :param zs:
        :return:
        """
        # Note there are no parameters we're being Bayesian about.The
        # parameters arr baked into how we specify the newural networks
        X,y = xs
        self.mapping(X)
        result = tf.exp(norm.logpdf(y,self.mus,self.sigma))
        result = tf.multiply(result,self.pi)
        result = tf.reduce_sum(result,1)
        result = tf.log(result)
        return tf.reduce_sum(result)





def sample_from_mixture(x,pred_weights,pred_means,pred_std,amount):
    """
    Draws samples from mixture model.
    Returns 2 d array with input X and sample from prediction of Mixture Model
    :param x:
    :param pred_weights:
    :param pred_means:
    :param pred_std:
    :param amount:
    :return:
    """
    samples = np.zeros((amount,2))
    n_mix = len(pred_weights[0])
    to_choose_from = np.arange(n_mix)
    for j,(weights,means,std_devs) in enumerate(zip(pred_weights,pred_means,pred_std)):
        index = np.random.choice(to_choose_from,p=weights)
        samples[j,1] = normal.rvs(means[index],std_devs[index],size=1)
        samples[j,0] = x[j]
        if j == amount-1:
            break
    return samples
