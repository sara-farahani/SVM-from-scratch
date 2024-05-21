import numpy as np
from cvxopt import matrix, solvers
from svm_primal import linear_svm_primal
from svm_dual import linear_svm_dual
from main import linear_svm_primal_via_dual
from dataset import load_split_dataset
from evaluate import calculate_metrics_and_plot
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
solvers.options['show_progress'] = False


def plot_results_2d(X, y, classifiers):
    # check whether data is 2d or not 
    if(X.shape[1] != 2):
        print("Data is not 2d!")
        exit()

    # set figure size
    plt.figure(figsize=(8, 6))

    # Plot decision boundaries
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    x0, x1 = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
    Z = np.c_[x0.ravel(), x1.ravel()]
    best_pred = find_best_prediction(Z, classifiers)
    Z = best_pred.reshape(x0.shape)
    plt.contourf(x0, x1, Z, cmap=plt.cm.Paired, alpha=0.8)

    # plot training points
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap="viridis")

    # set labels
    plt.title('Multiclass SVM')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.show()


def load_iris_data():
    #load iris dataset
    iris = load_iris()
    X, y = iris.data, iris.target
    # select two
    # X = iris.data[:, 1:3]
    X = iris.data[:, 2:4]
    # generate random numbers to shuffle dataset
    indices = np.random.randint(low=0, high=X.shape[0], size=X.shape[0])
    X = X[indices]
    y = y[indices]
    
    # split data into train and test data
    X_train, X_test, y_train, y_test = X[:100, :], X[100:, :], y[:100], y[100:]

    return X_train, X_test, y_train, y_test


def predict(X, w, b):
    # predict scores and classes
    y_pred = np.dot(X, w) - b
    return np.sign(y_pred), y_pred


def find_best_prediction(X, classifiers):
    # find the max score to find the max-margin 
    predictions = []
    for svm in classifiers:
        w = svm["weights"]
        b = svm["bias"]
        _, y_pred = predict(X, w, b)
        predictions.append(y_pred)
    
    best_pred = np.argmax(np.array(predictions), axis=0)

    return best_pred


def multiclass_svm(X, y, C=1):

    ovr_classifiers= []
    num_classes = len(np.unique(y_train))
    # one-vs-rest
    for i in range(num_classes):
        binary_y = y_train.copy()
        binary_y[binary_y !=  i] = -1
        binary_y[binary_y == i] = 1
        w_dual, b_dual = linear_svm_primal_via_dual(X_train, binary_y, C)
        ovr_classifiers.append({'weights':w_dual, 'bias':b_dual})

    return ovr_classifiers
    

if __name__== "__main__":
    
     # load iris dataset
    X_train, X_test, y_train, y_test = load_iris_data()
    
    C = 1
    
    classifiers = multiclass_svm(X_train, y_train, C)

    # find the max score to find the max-margin 
    best_pred = find_best_prediction(X_test, classifiers)
    # best_pred = find_best_prediction(X_train, classifiers)
    
    print("\nMulticlass Evaluation : ")
    calculate_metrics_and_plot(y_test, best_pred, labels=np.unique(y_test))
    # calculate_metrics_and_plot(y_train, best_pred, labels=np.unique(y_train))

    plot_results_2d(X_test, y_test, classifiers)
    