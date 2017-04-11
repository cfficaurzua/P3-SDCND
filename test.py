import numpy as np
import matplotlib.pyplot as plt

%matplotlib inline
n = 100
sigma = 0.1
x = np.linspace(-0.5, 0.5)
means = np.linspace(-0.5, 0.5,n)


def gaussmf(x,mean,sigma):
    return np.exp(-((x-mean)**2.)/float(sigma)**2.)

def distribute(x, sigma = 0.1, n = 10, width = (-0.5,0.5)):

    output = []
    for i in range(n):
        output.append(gaussmf(x,means[i],sigma))
    return np.array(output)


dis = distribute(.18,n=n)
print(means.shape, dis.shape)
plt.figure()
plt.bar(means,dis, width = 0.1, align='center')
def retrieve_steering_angle(neurons, width = (-0.5,0.5)):
    n = neurons.shape[0]
    means = np.linspace(width[0],width[1], n)
    output = np.dot(means,neurons)/np.sum(neurons)
    return output
print(retrieve_steering_angle(dis))
