import numpy as np
from math import exp
from sklearn.cross_validation import train_test_split
from sklearn.datasets import load_digits
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelBinarizer
from sklearn.cluster import KMeans
from numpy import genfromtxt


def tanh(x):
    return np.tanh(x)


def tanh_deriv(x):
    return 1.0 - np.tanh(x) ** 2


def logistic(x):
    return 1 / (1 + np.exp(-x))


def logistic_derivative(x):
    return logistic(x) * (1 - logistic(x))


def rbfnn(X, y, epochs, eta, X_test):
    X = np.atleast_2d(X)
    '''
    temp1 = np.ones([X.shape[0], X.shape[1]+1])
    temp1[:, 0:-1] = X  # adding the bias unit to the input layer
    X = temp1
    X_test = np.atleast_2d(X_test)
    temp2 = np.ones([X_test.shape[0], X_test.shape[1]+1])
    temp2[:, 0:-1] = X_test  # adding the bias unit to the input layer
    X_test = temp2
    '''
    y = np.array(y)
    c = len(set(y))
    nodes = 10 * c
    labels = LabelBinarizer().fit_transform(y)
    kmeans = KMeans(n_clusters=nodes, random_state=0).fit(X)
    centers = kmeans.cluster_centers_
    # centers = kmeans(X,nodes)
    # print(len(centers))

    layers = []
    layers.append(64)
    layers.append(nodes)
    layers.append(c)

    w = (2 * np.random.random((c, nodes)) - 1) * 0.25

    for iteration in range(epochs):
        i = np.random.randint(X.shape[0])
        current = X[i]
        temp = []
        X_new = []
        for j in range(len(centers)):
            # print(X[i]-centers[i])
            var = np.linalg.norm(current - centers[j])
            temp.append(exp(-1 * var * var / 2))

        X_new = np.array(temp)
        ans = np.dot(w, X_new)
        Y, der = [], []
        for p in ans:
            Y.append(logistic(p))
            der.append(logistic_derivative(p))

        Y = np.array(Y)
        der = np.array(der)
        error = Y - labels[i]
        k = 0
        # print(error,X_new[k],der, error*der)
        val = []
        for k in range(nodes):
            val.append(sum(w[:, k] * -1 * eta * error * X_new[k] * der))
            w[:, k] = w[:, k] - eta * error * X_new[k] * der

        for p in range(nodes):
            for j in range(64):
                centers[p][j] -= eta * val[p] * (current[j] - centers[p][j])
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
'''

def kmeans(data, k):
    centroids = []

    centroids = randomize_centroids(data, centroids, k)

    old_centroids = [[] for i in range(k)]

    iterations = 0
    while not (has_converged(centroids, old_centroids, iterations)):
        iterations += 1

        clusters = [[] for i in range(k)]

        # assign data points to clusters
        clusters = euclidean_dist(data, centroids, clusters)

        # recalculate centroids
        index = 0
        for cluster in clusters:
            old_centroids[index] = centroids[index]
            centroids[index] = np.mean(cluster, axis=0).tolist()
            index += 1

    # print("The total number of data instances is: " + str(len(data)))
    # print("The total number of iterations necessary is: " + str(iterations))
    # print("The means of each cluster are: " + str(centroids))

    print("The clusters are as follows:")
    for cluster in clusters:
        print("Cluster with a size of " + str(len(cluster)) + " starts here:")
        print(np.array(cluster).tolist())
        print("Cluster ends here.")
  
    # print(clusters[0])
    # clusters = np.array(clusters)
    # print(clusters.shape[1])

    variance = []
    for i in range(clusters.shape[1]):
        variance.append(np.var(clusters[i,:]))

    print(variance)
 
    return centroids


# Calculates euclidean distance between
# a data point and all the available cluster
# centroids.
def euclidean_dist(data, centroids, clusters):
    for instance in data:
        # Find which centroid is the closest
        # to the given data point.
        mu_index = min([(i[0], np.linalg.norm(instance - centroids[i[0]])) \
                        for i in enumerate(centroids)], key=lambda t: t[1])[0]
        try:
            clusters[mu_index].append(instance)
        except KeyError:
            clusters[mu_index] = [instance]

    # If any cluster is empty then assign one point
    # from data set randomly so as to not have empty
    # clusters and 0 means.
    for cluster in clusters:
        if not cluster:
            cluster.append(data[np.random.randint(0, len(data), size=1)].flatten().tolist())

    return clusters


# randomize initial centroids
def randomize_centroids(data, centroids, k):
    for cluster in range(0, k):
        centroids.append(data[np.random.randint(0, len(data), size=1)].flatten().tolist())
    return centroids


# check if clusters have converged
def has_converged(centroids, old_centroids, iterations):
    MAX_ITERATIONS = 100
    if iterations > MAX_ITERATIONS:
        return True
    return old_centroids == centroids

'''
def main():
    digits = load_digits()
    data = []
    X = []
    y = []
    my_data = genfromtxt('train.csv', delimiter=',')
    for i in range(1, len(my_data)):
        y.append(my_data[i][0])
        X.append(my_data[i][1:])
    X, y = np.array(X), np.array(y)
    X -= X.min()  # normalize the values to bring them into the range 0-1
    X /= X.max()
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    ans = rbfnn(X_train, y_train, 500, 0.3, X_test)
    predictions = []
    for o in ans:
        predictions.append(np.argmax(o))
    temp = confusion_matrix(y_test, predictions)
    print(temp)
    ans = np.array(temp)
    print('correctly matched patterns', ans.trace())
    print('total patterns', sum(sum(ans)))
    print('accuracy', ans.trace() * 1.0 / sum(sum(ans)))
    # print(classification_report(y_test,predictions))

main()