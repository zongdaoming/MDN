from scipy.stats import norm as normal
from scipy.stats import norm
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
import keras.backend as K
from keras.layers import Dense
from sklearn.model_selection import train_test_split
import numpy as np
import edward as ed
from edward.models import Normal,Categorical,Mixture


from sklearn.model_selection import train_test_split

def build_toy_dataset(N):
  y_data = np.random.uniform(-10.5, 10.5, N)
  r_data = np.random.normal(size=N)  # random noise
  x_data = np.sin(0.75 * y_data) * 7.0 + y_data * 0.5 + r_data * 1.0
  x_data = x_data.reshape((N, 1))
  return train_test_split(x_data, y_data, random_state=42)




N = 5000  # number of data points
D = 1  # number of features


X_train, X_test, y_train, y_test = build_toy_dataset(N)
print("Size of features in training data: {}".format(X_train.shape))
print("Size of output in training data: {}".format(y_train.shape))
print("Size of features in test data: {}".format(X_test.shape))
print("Size of output in test data: {}".format(y_test.shape))

def plot_scatter(data_x,data_y):
    from  matplotlib  import rc
    rc('font', **{'family': 'sans-serif', 'sans-serif': ['Times New Roman'], 'size': 14})
    params = {'text.usetex': False, 'mathtext.fontset': 'stixsans'}
    plt.rcParams.update(params)

    fig, ax = plt.subplots(num=1, figsize=(8, 8))
    plt.subplots_adjust(right=0.99, left=0.125, bottom=0.15, top=0.975)

    # ax.spines['right'].set_visible(False)
    # ax.spines['top'].set_visible(False)

    # ax.grid(True, linestyle = "--", color = "k", linewidth = "0.6")
    ax.xaxis.grid(True, which='major', lw=0.5, linestyle='--', color='0.3', zorder=1)
    ax.yaxis.grid(True, which='major', lw=0.5, linestyle='--', color='0.3', zorder=1)

    plt.scatter(data_x,data_y,s=None,c=None,marker=None,cmap=None,alpha=None,linewidths=None,edgecolors=None)
    plt.show()

# plot_scatter(X_train,y_train)


def neural_network(X):
    """loc, scale, logits = NN(x; theta)"""
    # 2 hidden layers with 15 hidden units
    net = tf.layers.dense(X, 15, activation=tf.nn.relu)
    net = tf.layers.dense(net, 15, activation=tf.nn.relu)
    locs = tf.layers.dense(net, 20, activation=None)
    scales = tf.layers.dense(net, 20, activation=tf.exp)
    logits = tf.layers.dense(net, 20, activation=None)
    return locs, scales, logits

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

def plot_normal_mix(pis, mus, sigmas, ax, label='', comp=True):
    """
    Plots the mixture of Normal models to axis=ax
    comp=True plots all components of mixtur model
    """
    x = np.linspace(-10.5, 10.5, 250)
    final = np.zeros_like(x)
    for i, (weight_mix, mu_mix, sigma_mix) in enumerate(zip(pis, mus, sigmas)):
        temp = normal.pdf(x, mu_mix, sigma_mix) * weight_mix
        final = final + temp
        if comp:
            ax.plot(x, temp, label='Normal ' + str(i))
    ax.plot(x, final, label='Mixture of Normals ' + label)
    ax.legend(fontsize=13)


X_ph = tf.placeholder(tf.float32, [None, D])
y_ph = tf.placeholder(tf.float32, [None])

locs, scales, logits = neural_network(X_ph)
cat = Categorical(logits=logits)
components = [Normal(loc=loc, scale=scale) for loc, scale
              in zip(tf.unstack(tf.transpose(locs)),
                     tf.unstack(tf.transpose(scales)))]
y = Mixture(cat=cat, components=components, value=tf.zeros_like(y_ph))

"""
Mixture API
def __init__(self,
               cat,
               components,
               validate_args=False,
               allow_nan_stats=True,
               use_static_graph=False,
               name="Mixture"):
"""


"""############################################################################
      Inference Model.We use MAP estimation,passing in the model and data set. 
   #############################################################################"""

inference = ed.MAP(data={y:y_ph}) # make the inference model

optimizer = tf.train.AdamOptimizer(5e-3)
inference.initialize(optimizer=optimizer, var_list=tf.trainable_variables())

sess = ed.get_session()
tf.global_variables_initializer().run()




n_epoch = 1000
train_loss = np.zeros(n_epoch)
test_loss = np.zeros(n_epoch)
for i in range(n_epoch):
  info_dict = inference.update(feed_dict={X_ph: X_train, y_ph: y_train})
  train_loss[i] = info_dict['loss']
  test_loss[i] = sess.run(inference.loss, feed_dict={X_ph: X_test, y_ph: y_test})
  inference.print_progress(info_dict)


pred_weights,pred_means,pred_std = sess.run([tf.nn.softmax(logits),locs,scales],feed_dict={X_ph:X_test})


fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(16, 3.5))
plt.plot(np.arange(n_epoch), -test_loss / len(X_test), label='Test')
plt.plot(np.arange(n_epoch), -train_loss / len(X_train), label='Train')
plt.legend(fontsize=20)
plt.xlabel('Epoch', fontsize=15)
plt.ylabel('Log-likelihood', fontsize=15)
plt.show()




# obj = [0, 4, 6]
# fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(16, 6))
#
# plot_normal_mix(pred_weights[obj][0], pred_means[obj][0], pred_std[obj][0], axes[0], comp=False)
# axes[0].axvline(x=y_test[obj][0], color='black', alpha=0.5)
#
# plot_normal_mix(pred_weights[obj][2], pred_means[obj][2], pred_std[obj][2], axes[1], comp=False)
# axes[1].axvline(x=y_test[obj][2], color='black', alpha=0.5)
#
# plot_normal_mix(pred_weights[obj][1], pred_means[obj][1], pred_std[obj][1], axes[2], comp=False)
# axes[2].axvline(x=y_test[obj][1], color='black', alpha=0.5)

# a = sample_from_mixture(X_test, pred_weights, pred_means, pred_std, amount=len(X_test))
# sns.jointplot(a[:,0], a[:,1], kind="hex", color="#4CB391", ylim=(-10,10), xlim=(-14,14))


"""
We define TensorFlow placeholders, which will be used to manually feed batches of data during inference. 
                  This is one of many ways to train models with data in Edward
"""

def build_toy_dataset(nsample=40000):
    y_data = np.float32(np.random.uniform(-10.5, 10.5, (1, nsample))).T
    r_data = np.float32(np.random.normal(size=(nsample,1))) # random noise
    x_data = np.float32(np.sin(0.75*y_data)*7.0+y_data*0.5+r_data*1.0)
    return train_test_split(x_data, y_data, random_state=42, train_size=0.1)

# X_train, X_test, y_train, y_test = build_toy_dataset()
# print("Size of features in training data: {}".format(X_train.shape))
# print("Size of output in training data: {}".format(y_train.shape))
# print("Size of features in test data: {}".format(X_test.shape))
# print("Size of output in test data: {}".format(y_test.shape))





# plot_scatter(X_train,y_train)


# sns.regplot(X_train, y_train, fit_reg=False)
# fig = plt.figure(figsize=(6,6))
# plt.plot(X_train,y_train,color="deepskyblue")


class MixtureDensityNetwork:
    """
    Mixture density network for outputs y on inputs x.
    p((x,y), (z,theta))
    = sum_{k=1}^K pi_k(x; theta) Normal(y; mu_k(x; theta), sigma_k(x; theta))
    where pi, mu, sigma are the output of a neural network taking x
    as input and with parameters theta. There are no latent variables
    z, which are hidden variables we aim to be Bayesian about.
    """
    def __init__(self, K):
        self.K = K # here K is the amount of Mixtures

    def mapping(self, X):
        """pi, mu, sigma = NN(x; theta)"""
        hidden1 = Dense(15, activation='relu')(X)  # fully-connected layer with 15 hidden units
        hidden2 = Dense(15, activation='relu')(hidden1)
        self.mus = Dense(self.K)(hidden2) # the means
        self.sigmas = Dense(self.K, activation=K.exp)(hidden2) # the variance
        self.pi = Dense(self.K, activation=K.softmax)(hidden2) # the mixture components

    def log_prob(self, xs, zs=None):
        """log p((xs,ys), (z,theta)) = sum_{n=1}^N log p((xs[n,:],ys[n]), theta)"""
        # Note there are no parameters we're being Bayesian about. The
        # parameters are baked into how we specify the neural networks.
        X, y = xs
        self.mapping(X)
        result = tf.exp(norm.logpdf(y, self.mus, self.sigmas))
        result = tf.multiply(result, self.pi)
        result = tf.reduce_sum(result, 1)
        result = tf.log(result)
        return tf.reduce_sum(result)



#
# ed.set_seed(42)
# model =MixtureDensityNetwork(20)
# X = tf.placeholder(tf.float32, shape=(None, 1))
# y = tf.placeholder(tf.float32, shape=(None, 1))
# data = ed.Data([X, y]) # Make Edward Data model
#
# inference = ed.MAP(model, data) # Make the inference model
# sess = tf.Session() # start TF session
# K.set_session(sess) # pass session info to Keras
# inference.initialize(sess=sess) # initialize all TF variables using the Edward interface
#
# pred_weights, pred_means, pred_std = sess.run([model.pi, model.mus, model.sigmas],
#                                               feed_dict={X: X_test})
#
# NEPOCH = 1000
# train_loss = np.zeros(NEPOCH)
# test_loss = np.zeros(NEPOCH)
# for i in range(NEPOCH):
#     _, train_loss[i] = sess.run([inference.train, inference.loss],
#                                 feed_dict={X: X_train, y: y_train})
#     test_loss[i] = sess.run(inference.loss, feed_dict={X: X_test, y: y_test})
#
# pred_weights, pred_means, pred_std = sess.run([model.pi, model.mus, model.sigmas],
#                                               feed_dict={X: X_test})
#
# fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(16, 3.5))
# plt.plot(np.arange(NEPOCH), test_loss/len(X_test), label='Test')
# plt.plot(np.arange(NEPOCH), train_loss/len(X_train), label='Train')
# plt.legend(fontsize=20)
# plt.xlabel('Epoch', fontsize=15)
# plt.ylabel('Log-likelihood', fontsize=15)
#
# obj = [0, 4, 6]
# fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(16, 6))
#
# plot_normal_mix(pred_weights[obj][0], pred_means[obj][0], pred_std[obj][0],
#                 axes[0], comp=False)
# axes[0].axvline(x=y_test[obj][0], color='black', alpha=0.5)
#
# plot_normal_mix(pred_weights[obj][2], pred_means[obj][2], pred_std[obj][2],
#                 axes[1], comp=False)
# axes[1].axvline(x=y_test[obj][2], color='black', alpha=0.5)
#
# plot_normal_mix(pred_weights[obj][1], pred_means[obj][1], pred_std[obj][1],
#                 axes[2], comp=False)
# axes[2].axvline(x=y_test[obj][1], color='black', alpha=0.5)
#
#
# a = sample_from_mixture(X_test, pred_weights, pred_means,
#                         pred_std, amount=len(X_test))
# sns.jointplot(a[:,0], a[:,1], kind="hex", color="#4CB391",
#               ylim=(-10,10), xlim=(-14,14))


