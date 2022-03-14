import numpy as np

#global size variables
n = 6
m = 4
sigma = 2

#Generating X
X = np.random.rand(n,m)

#Determine your beta parameters
beta = np.random.randn(m,1)

#Produce your noise vector epsilon
epsilon = np.random.randn(n,1)

#Compute Y
Y = np.matmul(X,beta) + epsilon

#Computing new beta
Xt = np.transpose(X)
sidea = np.linalg.inv(np.matmul(Xt,X))
sideb = np.matmul(Xt,Y)

new_beta = np.matmul(sidea,sideb)

#Compare the two betas
print("beta:\n",beta)
print("new beta:\n",new_beta)

#Increasing the sigma of the noise filter is done by mutiplying the vector by a constant sigma
#Taken from the numpy documentation
epsilon = sigma * epsilon

#Re-computing Y and re-estimate the beta paramaters
Y = np.matmul(X,beta) + epsilon
sidea = np.linalg.inv(np.matmul(Xt,X))
sideb = np.matmul(Xt,Y)

new_sigma_beta = np.matmul(sidea,sideb)


print("new beta after sigma increase:\n",new_sigma_beta)