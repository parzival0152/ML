import scipy.io
import numpy as np
from PIL import Image as im

mat = scipy.io.loadmat("EX2\mnist_all.mat")

# collage = im.new("L",(28*3,28*3))
# for i in range(9):
#     collage.paste(im.fromarray(np.array(mat.get(f'train{i+1}'))[0].reshape((28,28))),(28 * (i%3),28 * (i//3)))
# # collage.show()
# we have that each training set contains a 784 long vector that represents a handwritten digit
# the digit in the name of the training set corrosponds to the image in the figure

w = np.random.randn(784,1) # create a weights vector that is normally random distributed
w = np.ones((784,1)) # setting it to a ones vector to get the same results while debugging


x = np.array(mat.get('train4')[0])
# print(np.matmul(x,w))
print(x.pop())