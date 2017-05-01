import numpy as np
from math import exp
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelBinarizer
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits
from numpy import genfromtxt

np.random.seed(123)


# MA
def logistic(x):
    return 1 / (1 + np.exp(-x))


def logistic_derivative(x):
    return logistic(x) * (1 - logistic(x))


# implementation of Radial Basis function Neural Networks
# Working on smaller dataset of images
'''
A neural network will be craeted with 64, nodes, c(10) architecture.
'''
def rbfnn(X, y, epochs, eta, X_test):
    X = np.atleast_2d(X)
    y = np.array(y)
    c = len(set(y)) #c is 10 classes
    #nodes in the hidden layer
    nodes = 2*c
    # Converting the response variable into 2-d matrix that contains 0 and 1 representing multilabel classification.
    labels = LabelBinarizer().fit_transform(y)
    # Using the standard KMeans of sklearn.cluster. Own implementation is very slower but works
    #kmeans to iniatialize the weights from first to hidden layer(stored in centers array)
    kmeans = KMeans(n_clusters=nodes, random_state=0).fit(X)
    centers = kmeans.cluster_centers_
    
    #weights between hidden and output layer are initialized randomly
    w = (2 * np.random.random((c, nodes)) - 1) * 0.25

    '''loop for backpropagation: in every iteration a data point is 
    selected and the weigths are backpropaged according to the formula'''
    
    for iteration in range(epochs):
        i = np.random.randint(X.shape[0])
        current = X[i]
        temp = []
        X_new = []
        
        #forward propagation first layer to hidden layer and store it in X_new 
        for j in range(len(centers)):
            var = np.linalg.norm(current - centers[j])
            temp.append(exp(-1 * var * var / 2))
        X_new = np.array(temp)
        
        #forward propagation hidden layer to output layer and store it in Y
        #(after using transfer function)
        ans = np.dot(w, X_new)
        Y, der = [], []
        for p in ans:
            Y.append(logistic(p))
            der.append(logistic_derivative(p))
        
        #backpropagation output to hidden w gets changed
        Y = np.array(Y)
        der = np.array(der)
        error = Y - labels[i]
        k = 0
        val = []
        for k in range(nodes):
            val.append(sum(w[:, k] * -1 * eta * error * X_new[k] * der))
            w[:, k] = w[:, k] - eta * error * X_new[k] * der
        
        #backpropagation between hidden and input centers gets changed
        for p in range(nodes):
            for j in range(64):
                centers[p][j] -= eta * val[p] * (current[j] - centers[p][j])
    
    ''' end of backpropagation layer '''
    ''' Now calculate the output using forward pass'''
    res = []
    for data in X_test:
        temp = []
        X_new = []
        for j in range(len(centers)):
            var = np.linalg.norm(data - centers[j])
            temp.append(exp(-1 * var * var / 2))

        X_new = np.array(temp)
        ans = np.dot(w, X_new)
        Y, der = [], []
        for p in ans:
            Y.append(logistic(p))
            der.append(logistic_derivative(p))
        res.append(der)
    return res



def main():
    # Smaller dataset to start with
    digits = load_digits()
    X = digits.data
    Y = digits.target
    # Normalize the values to bring them into the range 0-1
    X -= X.min()  
    X /= X.max()
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y)
    #try changing iterations and learning rate
    ans = rbfnn(X_train, Y_train, 500, 0.3, X_test)
    predictions = []
    for o in ans:
        predictions.append(np.argmax(o))
    temp = confusion_matrix(Y_test, predictions)
    print(temp)
    ans = np.array(temp)
    print('correctly matched patterns', ans.trace())
    print('total patterns', sum(sum(ans)))
    print('accuracy', ans.trace() * 1.0 / sum(sum(ans)))

main()