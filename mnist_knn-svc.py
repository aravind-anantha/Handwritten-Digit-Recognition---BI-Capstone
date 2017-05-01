import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn import decomposition
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC, SVC

#Reading input data, the data is from the kaggle dataset on digit recognition
data = pd.read_csv('train.csv')
#Splitting the data into traina and tes, 80% and 20% respectively
train, test = np.split(data.sample(frac=1), [int(0.8*len(data))])

#train_label contains class labels, 0 to 9 since we are classifying digits
#Converting the pixels of the image to float for normalization
train_label = train['label']
train_image = (train.ix[:,1:].values).astype('float32')
test_label = test['label']
test_image = (test.ix[:,1:].values).astype('float32')

#Normalizing the data
train_image = train_image / 255
test_image = test_image / 255

#Checking to see how many components to select
pca = decomposition.PCA(n_components = 150)
pca.fit(train_image)
fig = plt.plot(pca.explained_variance_ratio_)
#plt.savefig('pca.png')

#Fitting a PCA model with 40 principal components
pca = decomposition.PCA(n_components = 40)
pca.fit(train_image)
train_pca = np.array(pca.transform(train_image))
pca.fit(test_image)
test_pca = np.array(pca.transform(test_image))


clf = KNeighborsClassifier(n_neighbors=4,algorithm='auto',n_jobs=10)
clf.fit(train_image, train_label)

predictions = clf.predict(test_image)
print(predictions)

print(confusion_matrix(test_label, predictions))
#array([[804,   0,   2,   0,   0,   1,   2,   0,   1,   1],
#       [  0, 921,   1,   0,   1,   0,   1,   2,   0,   2],
#       [  4,  10, 830,   0,   0,   0,   0,  14,   1,   0],
#       [  1,   2,   7, 858,   0,   6,   0,   4,   2,   3],
#       [  1,  13,   0,   0, 825,   0,   1,   1,   0,  17],
#       [  7,   0,   0,  14,   1, 711,   2,   0,   1,   0],
#       [  7,   3,   0,   0,   0,   6, 795,   0,   0,   0],
#       [  1,  11,   1,   0,   4,   0,   0, 856,   0,   6],
#       [  1,  14,   3,  14,   3,  11,   4,   3, 756,   4],
#      [  1,   0,   1,   6,  18,   2,   0,  23,   2, 769]])
print(accuracy_score(test_label, predictions))
#accuray = 0.9672

clf = SVC(C=12,
          kernel='rbf',
          class_weight='balanced',
          random_state=42
         )
clf.fit(train_image, train_label)

predictions = clf.predict(test_image)
predictions

print(confusion_matrix(test_label, predictions))
#[[845   0   1   1   0   0   7   0   2   1]
# [  0 957   2   2   1   0   0   1   2   0]
# [  2   3 794   4   7   1   4   7   5   1]
# [  2   6  14 823   1  17   3   3   3   3]
# [  2   1   2   0 770   0   5   1   2  24]
# [  6   3   3  16   1 696   6   0   5   2]
# [  1   0   2   0   2   6 764   0   2   0]
# [  1   8  14   2   6   0   0 872   3  11]
# [  3  13   4  16   5  13   3   3 726   4]
# [  3   3   3  11  21   4   1  16   4 780]]
print(accuracy_score(test_label, predictions))
#accuracy = 0.9555

#Adding a Random Forest Model
lf = RandomForestClassifier(n_estimators=250,
                              n_jobs=-1,
                              random_state=42,
                              class_weight='balanced',
                              criterion='gini'
 )

clf.fit(train_image, train_label)

predictions = clf.predict(test_image)
print(predictions)

print(confusion_matrix(test_label, predictions))
# [  0 968   4   5   3   1   0   1   0   1]
# [  6   0 785   3   5   0   3  17   5   1]
# [  2   1   8 773   1   7   2   7  12   9]
# [  0   2   0   1 810   0   2   1   1  18]
# [  0   0   0  15   0 683   2   0   4   5]
# [  6   2   0   0   2   8 833   0   4   0]
# [  1   2   4   2   3   1   0 878   1   7]
# [  3   4   1   7   4   6   3   0 784  12]
# [  3   1   0  10   7   2   0   5   2 787]]

print(accuracy_score(test_label, predictions))
#accuracy = 0.9669