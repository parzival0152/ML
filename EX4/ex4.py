import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import multivariate_normal
# creating the data
num_data = 1000

a = [0.5, 0.2, 0.3]

sigma1 = [[1.2, 0], [0, 1.2]]
sigma2 = [[1.1, 0], [0, 1.1]]
sigma3 = [[1.3, 0], [0, 1.3]]

mu1 = [-2, 5]
mu2 = [2.5, 2]
mu3 = [-2, 0.5]


distributions = [np.random.multivariate_normal(mu1, sigma1, num_data), np.random.multivariate_normal(
    mu2, sigma2, num_data), np.random.multivariate_normal(mu3, sigma3, num_data)]

draw = np.random.choice([1, 2, 3], num_data, p=a)
data = []
for i in range(num_data):
    data.append(distributions[draw[i]-1][i])

d1 = [data[i] for i in np.where(draw == 1)[0].tolist()]
d2 = [data[i] for i in np.where(draw == 2)[0].tolist()]
d3 = [data[i] for i in np.where(draw == 3)[0].tolist()]
# ploting the data
x1, y1 = np.array(d1).T
x2, y2 = np.array(d2).T
x3, y3 = np.array(d3).T
sizes1 = np.ones(len(d1))*10
sizes2 = np.ones(len(d2))*10
sizes3 = np.ones(len(d3))*10
plt.scatter(x1, y1, s=sizes1)
plt.scatter(x2, y2, s=sizes2)
plt.scatter(x3, y3, s=sizes3)

# K-means:
# choosing somethng close to the real center
center1 = [-2, 3]
center2 = [2, 1]
center3 = [-2, 0]

for i in range(20):
    c1 = []
    c2 = []
    c3 = []
    dataLkMeans = []
    for i in range(num_data):
        l = [np.linalg.norm(data[i]-center1), np.linalg.norm(data[i] - center2), np.linalg.norm(data[i]-center3)]
        if l.index(min(l)) == 0:
            c1.append(data[i])
        elif l.index(min(l)) == 1:
            c2.append(data[i])
        elif l.index(min(l)) == 2:
            c3.append(data[i])
        dataLkMeans.append(l.index(min(l))+1)
    # update center
    center1 = sum(c1)/len(c1)
    center2 = sum(c2)/len(c2)
    center3 = sum(c3)/len(c3)
# check success rate of K-means
checkMistakes = 0
for i in range(num_data):
    if draw[i] != dataLkMeans[i]:
        checkMistakes += 1
mistakesRate = 1-(checkMistakes/num_data)
print("K-means:")
print("number of mistakes", checkMistakes)
print("success rate of K-means", mistakesRate)

# GMM
# initial parameters:

phi1 = (1/3)
phi2 = (1/3)
phi3 = (1/3)

muj1 = [-2, 4]
muj2 = [2, 2]
muj3 = [-2, 0]

sigmaj1 = [[1.1, 0], [0, 1.1]]
sigmaj2 = [[1.1, 0], [0, 1.1]]
sigmaj3 = [[1.1, 0], [0, 1.1]]
data = np.array(data)
flag = True
for i in range(50):
    if flag:
        mvn1 = multivariate_normal(muj1, sigmaj1).pdf(data)*phi1
        mvn2 = multivariate_normal(muj2, sigmaj2).pdf(data)*phi2
        mvn3 = multivariate_normal(muj3, sigmaj3).pdf(data)*phi3
        n = mvn1+mvn2+mvn3
        wt1 = np.array(mvn1/n)
        wt2 = np.array(mvn2/n)
        wt3 = np.array(mvn3/n)
    if not flag:
        sumwt1 = sum(wt1)
        sumwt2 = sum(wt2)
        sumwt3 = sum(wt3)
        phi1 = sumwt1/num_data
        phi2 = sumwt2/num_data
        phi3 = sumwt3/num_data
        muj1 = sum(data*wt1[:, None])/sumwt1
        muj2 = sum(data*wt2[:, None])/sumwt2
        muj3 = sum(data*wt3[:, None])/sumwt3
        sigmaj1 = sum((data-muj1)*(data-muj1)*wt1[:, None])/sumwt1
        sigmaj2 = sum((data-muj2)*(data-muj2)*wt2[:, None])/sumwt2
        sigmaj3 = sum((data-muj3)*(data-muj3)*wt3[:, None])/sumwt3
    flag != flag
lGMM = []
for i in range(num_data):
    l = [wt1[i], wt2[i], wt3[i]]
    lGMM.append(l.index(max(l))+1)
# check success rate of GMM
checkMistakes = 0
for i in range(num_data):
    if draw[i] != lGMM[i]:
        checkMistakes += 1
mistakesRate = 1-(checkMistakes/num_data)
print("GMM:")
print("number of mistakes", checkMistakes)
print("success rate of GMM", mistakesRate)

plt.show()
