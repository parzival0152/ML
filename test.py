import numpy as np
from scipy.spatial import distance_matrix

vectors = np.array([
    [1,0],
    [2,0],
    [3,0]
])

points = np.array([
    [4,0],
    [2,0],
    [3,0]
])

# print(np.linalg.norm(vectors-points,axis=1))
dm= distance_matrix(vectors,points)
centers = np.argmin(dm,axis=1)
print(vectors)
print((centers==1).reshape((3,1)))
print(np.sum(vectors,where=(centers==1).reshape((3,1)),axis=0))