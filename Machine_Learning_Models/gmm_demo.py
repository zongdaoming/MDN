
import numpy as np

def generator_data(k,mu,sigma,N):
    """
    generate gmmm data
    :param k: 混合系数
    :param mu: 均值
    :param sigma: 协方差
    :param N: number of data
    :return: 生成的数据
    """
    # initialize the data
    dataArray = np.zeros(N,dtype=np.float32)
    # Generate data according to the probability.
    # number of gaussian distribution

    n = len(k)
    for i in range(N):
        rand = np.random.random()
        sum = 0
        index = 0
        while(index < n):
            sum += k[index]
            if (rand <sum ):
                dataArray[i] = np.random.normal(mu[index],sigma[index])
                break
            else:
                index += 1
    return dataArray


def normPDF(x,mu,sigma):
    """
    计算均值为mu,标准差为sigma的正态分布函数的密度函数值
    :param x:
    :param mu:
    :param sigma:
    :return: x处的概率密度值
    """
    return (1./np.sqrt(2*np.pi))*(np.exp(-(x-mu)**2/(2*sigma**2)))

def em(dataArray,k,mu,sigma,step=10):
    """
    EM 算法估计高斯过程
    :param dataArray: 已知数据个数
    :param k: 每个高斯分布的估计系数
    :param mu: 每个高斯分布的估计均值
    :param sigma: 每个高斯分布的估计标准差
    :param step: 迭代次数
    :return: em 估计迭代结束估计的参数值【k,mu,sigma】
    """
    # number of gaussian distribution
    n = len(k)
    # number of data
    N =  dataArray.size
    gammaArray = np.zeros((n,N))
    for s in range(step):
        for i in range(n):
            for j in range(N):
                SUM = sum([k[t] * normPDF(dataArray[j],mu[t],sigma[t]) for t in range(n)])
                gammaArray[i][j] = k[i]*normPDF(dataArray[j],mu[i],sigma[i])/float(SUM)
        # update  mu
        for i in range(n):
            mu[i] = np.sum(gammaArray[i]*dataArray)/np.sum(gammaArray[i])
        # update sigma
        for i in range(n):
            sigma[i] = np.sqrt(np.sum(gammaArray[i]*(dataArray-mu[i])**2)/np.sum(gammaArray[i]))
        # update coefficient
        for i in range(n):
            k[i] = np.sum(gammaArray[i])/N
    return [k,mu,sigma]

if __name__ == '__main__':
    # 参数的准确值
    k = [0.3,0.4,0.3]
    mu = [2,4,3]
    sigma = [1,1,4]
    # 样本数
    N = 50000
    # Generate data
    dataArray = generator_data(k,mu,sigma,N)
    # 参数的初始值，注意em算法对于参数的初始值是十分敏感的
    k0 = [0.3,0.3,0.4]
    mu0 = [1,2,2]
    sigma0 = [1,1,1]
    step = 6
    # 使用EM算法计算参数
    k1,mu1,sigma1 = em(dataArray,k0,mu0,sigma0,step)
    # 输出参数值
    print("参数实际值:")
    print("k:", k)
    print("mu:", mu)
    print("sigma:", sigma)
    print("参数估计值:")
    print("k1:", k1)
    print("mu1:", mu1)
    print("sigma1:", sigma1)


