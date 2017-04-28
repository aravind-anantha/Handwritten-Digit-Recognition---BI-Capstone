import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn import decomposition
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC, SVC
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
from keras.optimizers import Adam
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score


data = pd.read_csv('train.csv')

train, test = np.split(data.sample(frac=1), [int(0.8*len(data))])

train_label = train['label']
train_image = (train.ix[:,1:].values).astype('float32')

test_label = test['label']
test_image = (test.ix[:,1:].values).astype('float32')

train_image = train_image / 255
test_image = test_image / 255



def elastic_deformation(image, alpha, sigma):
    random_state = np.random.RandomState(None)
    shape = image.shape
    
    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    
    x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')
    indices = np.reshape(x+dx, (-1, 1)), np.reshape(y+dy, (-1, 1))
    
    return map_coordinates(image, indices, order=1).reshape(shape)

t = train_image.reshape(train_image.shape[0], 28, 28)

for i in range(len(train_image)):
    img = elastic_deformation(t[i], 36, 8)
    img = img[np.newaxis, :, :]
    t = np.concatenate((t, img))

labels = train_label.append(train_label[:], ignore_index = True)

for i in range(6, 9):
    plt.subplot(330 + (i+1))
    plt.imshow(t[i], cmap=plt.get_cmap('gray'))
    plt.title(labels[i]);

plt.savefig('before_deformation.png')

for i in range(33606, 33609):
    plt.subplot(330 + (i+1) - 33600)
    plt.imshow(t[i], cmap=plt.get_cmap('gray'))
    plt.title(labels[i]);

plt.savefig('after_deformation.png')

temp = t.reshape(t.shape[0], -1)

clf = SVC(C=12,
          kernel='rbf',
          class_weight='balanced',
          random_state=42
         )
clf.fit(temp, labels)

predictions = clf.predict(test_image)
predictions

print(confusion_matrix(test_label, predictions))
#[[823   0   0   1   0   1   2   0   3   1]
# [  0 968   6   1   2   0   0   1   1   4]
# [  4   0 792   2   3   1   5  11   5   2]
# [  0   4   6 773   1  13   2   6  11   6]
# [  0   3   1   0 809   2   4   1   0  15]
# [  0   3   2  16   2 679   4   1   0   2]
# [  5   2   0   0   1   9 837   0   1   0]
# [  2   3   5   1   5   2   0 872   0   9]
# [  4   7   6  11   7  11   3   2 767   6]
# [  3   4   1   8  10   1   0   5   2 783]]
print(accuracy_score(test_label, predictions))
#accuracy = 0.9646

clf = KNeighborsClassifier(n_neighbors=4,algorithm='auto',n_jobs=10)
clf.fit(temp, labels)

predictions = clf.predict(test_image)
print(predictions)

print(confusion_matrix(test_label, predictions))
#[[827   0   0   2   1   0   1   0   0   0]
# [  0 977   3   0   1   0   0   0   0   2]
# [  4   9 790   1   0   0   0  17   3   1]
# [  2   5   5 789   1   7   0   2   7   4]
# [  0   8   0   0 811   0   2   2   0  12]
# [  1   0   0  13   1 683   5   0   2   4]
# [  5   4   0   0   0   8 838   0   0   0]
# [  1   6   6   0   2   0   0 879   0   5]
# [  4  10   3  19   6  10   4   3 761   4]
# [  4   1   1   7   7   1   0   8   1 787]]
print(accuracy_score(test_label, predictions))
#accuray = 0.9692

train_x = temp[:,:].reshape(temp.shape[0], 28, 28, 1).astype('float32')
train_y = to_categorical(labels, 10)

test_x = test_image[:,:].reshape(test_image.shape[0], 28, 28, 1).astype('float32')


from keras import backend

backend.image_data_format()


model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=(28, 28, 1)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

model_fit = model.fit(train_x, train_y, batch_size=128, epochs=12)

predictions = model.predict_classes(test_x)

print(confusion_matrix(test_label, predictions))
#[[826   0   1   0   0   0   2   0   1   1]
# [  0 979   2   0   0   0   0   2   0   0]
# [  0   0 815   1   0   0   0   7   0   2]
# [  0   0   1 813   0   2   0   3   0   3]
# [  1   2   1   0 824   0   2   2   0   3]
# [  0   0   0   2   0 703   1   0   2   1]
# [  2   1   0   1   1   2 848   0   0   0]
# [  0   0   5   0   1   0   0 892   0   1]
# [  2   0   0   0   1   1   0   0 818   2]
# [  1   1   0   0   2   0   0   1   1 811]]
print(accuracy_score(test_label, predictions))
#accuray = 0.9915



