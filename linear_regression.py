import numpy as np
from skimage.feature import hog
from tqdm import trange

def hypothesis(theta, X, n):
    h = np.ones((X.shape[0],1))
    theta = theta.reshape(1,n+1)
    for i in range(0,X.shape[0]):
        h[i] = float(np.matmul(theta, X[i]))
    h = h.reshape(X.shape[0])
    return h

def BGD(theta, alpha, num_iters, h, X, y, n):
    llambda = 10
    cost = np.ones(num_iters)
    for i in trange(0,num_iters):
        theta[0] = theta[0] - (alpha/X.shape[0]) * sum(h - y)
        for j in range(1,n+1):
            theta[j] = theta[j] - (alpha/X.shape[0]) * sum((h-y) * X.transpose()[j])
        h = hypothesis(theta, X, n)
        cost[i] = (1/X.shape[0]) * 0.5 * sum(np.square(h - y)) + (llambda * np.sum(np.square(theta)))
        print(cost)
#         if(cost[i] < 0.003):
#             break
    theta = theta.reshape(1,n+1)
    return theta, cost

def linear_regression(X, y, alpha, num_iters):
    n = X.shape[1]
    one_column = np.ones((X.shape[0],1))
    X = np.concatenate((one_column, X), axis = 1)
    # initializing the parameter vector...
    theta = np.zeros(n+1)
    # hypothesis calculation....
    h = hypothesis(theta, X, n)
    # returning the optimized parameters by Gradient Descent...
    theta, cost = BGD(theta,alpha,num_iters,h,X,y,n)
    return theta, cost

def get_predictor(X, y, iterations, feature_id):
    alpha = 0.001
    theta, cost = linear_regression(X, y[:, feature_id], alpha, iterations)
    return theta

def cut_data(X_train, y_train):
    X_new = X_train.reshape(X_train.shape[0], 96, 96)
    X_new = np.array(list(map(lambda image: hog(image, orientations=9, pixels_per_cell=(8, 8),
                    cells_per_block=(1, 1), visualize=False, multichannel=False), X_new)))
#     X_new = np.array(list(map(lambda image: hog(image), X_new)))
    X = (X_new - X_new.mean()) / X_new.std()
    y = (y_train - y_train.mean()) / y_train.std()
#     X, y = X_new, y_train
    return X, y

class Predictor:
    iterations = 6

    def __init__(self, X, y):
        self.thetas = []
        self.features = list(range(30))
#         self.features = [0, 1, 2, 3, 20, 21, 26, 27]
#         self.features = [0]
        X_new, y_new = cut_data(X, y)
        self.y = y
        self.X, _ = X_new, y_new
        for i in self.features:
            print("Processing keypoint", i)
            self.thetas.append(get_predictor(X_new, y_new, Predictor.iterations, i))

    def predict(self, X, feature_id):
        X_new = np.array(list(map(lambda image: hog(image, orientations=9, pixels_per_cell=(8, 8),
                cells_per_block=(1, 1), visualize=False, multichannel=False), X)))
#         X_new = np.array(list(map(lambda image: hog(image), X_new)))
#         X_new = (X_new - X_new.mean()) / X_new.std()к
        X_1 = np.concatenate((np.ones((X_new.shape[0], 1)), X_new), axis = 1)
        return np.dot(self.thetas[self.features.index(feature_id)], X_1.T) * self.y.std() + self.y.mean()

    def predict_all(self, X):
        X_new = np.array(list(map(lambda image: hog(image, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(1, 1), visualize=False, multichannel=False), X)))
#         X_new = np.array(list(map(lambda image: hog(image), X_new)))
        X_new = (X_new - X_new.mean()) / X_new.std()
        X_1 = np.concatenate((np.ones((X_new.shape[0], 1)), X_new), axis = 1)
        y_res = []
        for keypoint in self.features:
            y_res.append(np.dot(self.thetas[keypoint], X_1.T) * self.y.std() + self.y.mean())
        return np.array(y_res).reshape(len(self.features), len(X))
