from PIL import Image
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import numpy as np
import os


#img = Image.open('data/images/soccer_ball/image_0001.jpg').convert('RGBA')
#arr = np.array(img)

# record the original shape
#shape = arr.shape

# make a 1-dimensional view of arr
#flat_arr = arr.ravel()
#print(len(flat_arr))

# convert it to a matrix
#vector = np.matrix(flat_arr)

# do something to the vector
#vector[:,::10] = 128

# reform a numpy array of the original shape
#arr2 = np.asarray(vector).reshape(shape)

# make a PIL image
#img2 = Image.fromarray(arr2, 'RGBA')
#img2.show()

dir_pizza = 'data/images/pizza'
dir_socc = 'data/images/soccer_ball'

training_data = []
training_labels = []
shapes = []

for file in os.listdir(dir_pizza):
    filename = os.fsdecode(file)
    img = Image.open(os.path.join(dir_pizza, filename)).convert('RGBA')
    arr = np.array(img)

    shapes.append(arr.shape)
    training_data.append(arr.ravel()[:180000])
    training_labels.append(-1)

for file in os.listdir(dir_socc):
    filename = os.fsdecode(file)
    img = Image.open(os.path.join(dir_socc, filename)).convert('RGBA')
    arr = np.array(img)

    shapes.append(arr.shape)
    training_data.append(arr.ravel()[:180000])
    training_labels.append(1)

svc_ = SVC(C=1, kernel='linear')
svc_.fit(training_data, training_labels)
err = 1 - accuracy_score(training_labels, svc_.predict(training_data))
print('err= ',err)