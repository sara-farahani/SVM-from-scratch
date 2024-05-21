import numpy as np
from cvxopt import matrix, solvers
import matplotlib
from svm_primal import linear_svm_primal
from svm_dual import linear_svm_dual
from dataset import load_split_dataset
from evaluate import calculate_metrics_and_plot
from visualize import visualize_binary_classification
import matplotlib.pyplot as plt
solvers.options['show_progress'] = False


def strong_duality(X, y, C=1):
    # find Lagrange multipliers from dual problem
    alpha = linear_svm_dual(X, y, C)

    # find weights and bias from primal problem
    w, b, distances = linear_svm_primal(X, y, C)

    # compute primal problem optimal value
    primal_optimal_value = np.dot(w.T, w) + C*np.sum(distances)

    # compute dual problem optimal value
    linear_kernel = X*np.reshape(y, (-1,1))
    P = np.dot(linear_kernel, linear_kernel.T) 
    dual_optimal_value = -0.25*(np.dot(np.dot(alpha.T, P), alpha)) + np.sum(alpha)

    print("\nStrong Duality : ")
    print("primal_optimal_value", float(primal_optimal_value))
    print("dual_optimal_value", float(dual_optimal_value))
    
    return


def linear_svm_primal_via_dual(X, y, C=1):
    # find Lagrange multipliers from dual problem
    alpha = linear_svm_dual(X, y, C)

    # find support vectors
    support_vectors = np.where((alpha > 1e-4) & (alpha <= C))[0]
    # compute weights using dual problem results
    w_dual = 0.5* np.dot((y[support_vectors]*alpha[support_vectors]).T,
                         X[support_vectors, :])
    # compute bias using dual problem results
    b_dual = np.mean(X[support_vectors, :].dot(w_dual) - y[support_vectors])
    
    # compute weights and bias directly from primal problem
    w_primal, b_primal, _ = linear_svm_primal(X, y, C=1)

    # reshape and print weights to check whether they are the same or not
    w_primal = np.reshape(w_primal, (-1))
    w_dual = np.reshape(w_dual, (-1))
    print("\nPrimal problem weights : ", w_primal, "bias : ", b_primal)
    print("Dual problem weights : ", w_dual, "bias : ", b_dual)

    return w_dual, b_dual


def predict(X, w, b):
    # predict scores and classes
    y_pred = np.dot(X, w) - b
    return np.sign(y_pred)


def plot_results_2d(X, y, w, b):
    if(X.shape[1] != 2):
        raise Exception("Data is not 2d!")
    # set figure size
    plt.figure(figsize=(8, 6))
    # plot data points
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', s=50)
    
    # plot decision boundaries
    x0, x1 = np.meshgrid(np.linspace(X[:, 0].min()-2, X[:, 0].max()+2, 50), np.linspace(X[:, 1].min()-2, X[:, 1].max()+2, 50))
    y = (w[0] * x0 + w[1] * x1 - b)
    plt.contour(x0, x1, y, colors='r', levels=[-1, 0, 1], alpha=0.5, linestyles=['--', '-', '--'])
    
    # set labels
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('SVM')
    plt.show()


def load_2d_datasets():
    dataset = "Datasets/with slack"
    # dataset = "Datasets/without slack"
    # dataset = "Datasets/not linearly separable"
    dataset = np.load("%s.npy" % dataset)
    X = dataset[:, :2] # feature points
    y = dataset[:, 2] # ground truth labels
    y[y==0] = -1
    # shuffle dataset
    indices = np.random.randint(low=0, high=X.shape[0], size=X.shape[0])
    X = X[indices]
    y = y[indices]
    split = int(X.shape[0] * 0.9)
    X_train, X_test, y_train, y_test = X[:split, :], X[split:, :], y[:split], y[split:]

    return X_train, X_test, y_train, y_test


def svm():

    # load Iris dataset
    X_train, X_test, y_train, y_test = load_split_dataset()
    ## convert label = 0 to label = -1
    y_train[y_train==0] = -1
    y_test[y_test==0] = -1

    # load 2d datasets
    # X_train, X_test, y_train, y_test = load_2d_datasets()
    # X_test, y_test = X_train, y_train
    
    C = 1
    # find weights and bias by solving primal problem
    w_primal, b_primal, distances = linear_svm_primal(X_train, y_train, C)
    print("\nlinear_svm_primal results: ", "\nweights : ", w_primal[:, 0], "\nbias : ", b_primal)

    # find Lagrange multipliers from dual problem
    alphas = linear_svm_dual(X_train, y_train, C)
    # print("\nlinear_svm_dual results:\n", "alphas : ", alphas)

    # check wether strong duality holds or not
    strong_duality(X_train, y_train, C)

    # find weights and bias from dual problem 
    w_dual, b_dual = linear_svm_primal_via_dual(X_train, y_train, C)
    print("\nlinear_svm_primal_via_dual results", "\nweights", w_dual, "\nbias", b_dual)
    
    # evaluate results of primal problem
    print("\nPrimal problem Evaluation : ")
    y_pred = predict(X_test, w_primal, b_primal)[:, 0]
    calculate_metrics_and_plot(y_test, y_pred, labels=np.unique(y_test))
    y_pred[y_pred==-1] = 0
    visualize_binary_classification(X_test, y_pred, y_pred, class_labels=np.unique(y_test))

    # evaluate results of dual problem
    print("\nDual via Dual Evaluation : ")
    y_pred = predict(X_test, w_dual, b_dual)
    calculate_metrics_and_plot(y_test, y_pred, labels=np.unique(y_test))
    y_pred[y_pred==-1] = 0
    visualize_binary_classification(X_test, y_pred, y_pred, class_labels=np.unique(y_test))

    # plot results for 2d data
    # plot_results_2d(X_test, y_test, w_dual, b_dual)

if __name__== "__main__":
    svm()
    
