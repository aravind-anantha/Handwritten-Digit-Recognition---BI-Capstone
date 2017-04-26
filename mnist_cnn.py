import numpy as np
import pandas as pd
import random

from keras.optimizers import Adam
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score


data = pd.read_csv('train.csv')

train, test = np.split(data.sample(frac=1), [int(0.8*len(data))])

train_x = train.values[:,1:].reshape(train.shape[0], 28, 28, 1).astype('float32') / 255
train_y = to_categorical(train.values[:, 0], 10)

test_x = test.values[:,1:].reshape(test.shape[0], 28, 28, 1).astype('float32') / 255
test_y = test.values[:,0]

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

confusion_matrix(test_y, predictions)
#array([[820,   0,   2,   0,   0,   0,   4,   0,   2,   0],
#       [  0, 934,   5,   0,   1,   1,   0,   3,   1,   0],
#       [  1,   0, 818,   2,   1,   0,   0,   5,   2,   1],
#       [  0,   0,   4, 862,   0,   5,   0,   1,   4,   2],
#       [  0,   0,   0,   0, 823,   0,   1,   1,   0,   5],
#       [  0,   0,   0,   5,   0, 694,   2,   0,   3,   3],
#       [  1,   0,   0,   1,   2,   2, 845,   0,   2,   0],
#       [  0,   1,   6,   0,   1,   0,   0, 876,   1,   1],
#       [  2,   0,   2,   0,   2,   0,   0,   1, 831,   5],
#       [  0,   0,   0,   0,   4,   1,   0,   2,   0, 793]])


accuracy_score(test_y, predictions)
#accuracy achieved is 0.9876


