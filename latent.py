import numpy as np
import pandas as pd
from collections import defaultdict
import matplotlib.pyplot as plt
from matplotlib import style

path = '/content/recolab-data/latent/train.txt'
train_data = []
test_data = []
with open(path) as f:
    data = f.read().split('\n')
    for line in data[:-1]:
        l = line.split('\t')

        train_data.append([int(x) for x in l])

train_data = np.array(train_data)
print(train_data.shape)

path = '/content/recolab-data/latent/test.txt'
with open(path) as f:
    data = f.read().split('\n')
    for line in data[:-1]:
        l = line.split('\t')

        test_data.append([int(x) for x in l])

test_data = np.array(test_data)
print(test_data.shape)

class Latent_Factor():
    def __init__(self,R,K,alpha,beta,iterations):
        self.R = R
        self.num_users, self.num_items = R.shape
        self.K = K
        self.alpha = alpha
        self.beta = beta
        self.iterations = iterations

    def train(self):
        # Initialize user and item latent feature matrice
        print("Latent factor : ",self.K)
        self.P = np.random.normal(scale=1./self.K, size=(self.num_users, self.K))
        self.Q = np.random.normal(scale=1./self.K, size=(self.num_items, self.K))
        

        # Create a list of training samples
        self.samples = [
            (i, j, self.R[i, j])
            for i in range(self.num_users)
            for j in range(self.num_items)
            if self.R[i, j] > 0
        ]

        training_process = []
        for i in range(self.iterations):
            np.random.shuffle(self.samples)
            self.sgd()
            mse = self.mse()
            training_process.append((i, mse))
            if (i+1) % 10 == 0:
                print("Iteration: %d ; error = %.4f" % (i+1, mse))

        return training_process
    
    def mse(self):
         xs, ys = self.R.nonzero()
         predicted = self.full_matrix()
         error = 0
         for x, y in zip(xs, ys):
             error += pow(self.R[x, y] - predicted[x, y], 2)
         return np.sqrt(error)

    def sgd(self):
        for i, j, r in self.samples:
            prediction = self.get_rating(i, j)
            e = 2*(r - prediction)

            # Update user and item latent feature matrices
            self.P[i, :] += self.alpha * (e * self.Q[j, :] - 2*self.beta * self.P[i,:])
            self.Q[j, :] += self.alpha * (e * self.P[i, :] - 2*self.beta * self.Q[j,:])

    def get_rating(self, i, j):
        prediction = self.P[i, :].dot(self.Q[j, :].T)
        return prediction

    def full_matrix(self): 
        return self.P.dot(self.Q.T)


Z = np.max(train_data,axis = 0)
n_users = Z[1]
n_items = Z[0]
print(n_users,n_items)

R = np.zeros((n_users,n_items))

for l in train_data:
    R[l[1]-1,l[0]-1] = l[2]

 

print(R[:5]) 
factors = [10, 20, 40, 80, 100, 200]
for k in factors:
    alpha = 0.01
    beta = 0.005
    lf = Latent_Factor(R, K=k, alpha=alpha, beta=beta, iterations=40)
    X = lf.train()
    test_error = 0.0
    predicted = lf.full_matrix()
    for l in test_data:
        test_error += pow(l[2] - predicted[l[1]-1, l[0]-1], 2)
    print('Test Error : ',test_error)
    X = np.array(X)
    style.use('ggplot')
    plt.figure(figsize=(12,6))
    plt.xlabel("Iteration")
    plt.ylabel("Mean Square Error")
    s = "Latent Factor : "+str(k)+" lr = "+str(alpha) + ", lambda = " + str(beta)
    plt.title(s)
    plt.plot(X[:,0],X[:,1])
    plt.show()


    

