from PIL import Image
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import numpy as np
import os

dir_tr = 'data/images/cat_dog/train'
dir_test = 'data/images/cat_dog/test'

training_data = []
training_labels = []
shapes = []
nit = 0
nit_test = 0
size = 64, 64


for file in os.listdir(dir_tr):
    filename = os.fsdecode(file)
    img = Image.open(os.path.join(dir_tr, filename)).convert('RGBA')
    img = img.resize(size, Image.ANTIALIAS)
    arr = np.array(img)

    #print(len(arr.ravel()))
    shapes.append(arr.shape)
    training_data.append(arr.ravel())

    if 'dog' in file:
        training_labels.append(-1.0)
    else:
        training_labels.append(1.0)

    nit += 1
    #print(nit)
    if nit>2000:
        break


print('reading done')
svc_ = SVC(C=1, kernel='linear')
svc_.fit(training_data, training_labels)
err = 1 - accuracy_score(training_labels, svc_.predict(training_data))
print(sum(training_labels))
print('err= ', err)

for file in os.listdir(dir_test):
    filename = os.fsdecode(file)
    img = Image.open(os.path.join(dir_test, filename)).convert('RGBA')
    img = img.resize(size, Image.ANTIALIAS)
    arr = np.array(img)
    label = svc_.predict([arr.ravel()])
    l = 'dog' if label[0]==-1.0 else 'cat'
    print('file:', filename, ' is: ',l)
    nit_test +=1
    if nit_test>0:
        break
