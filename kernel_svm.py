import numpy as np
from cvxopt import matrix, solvers
from svm_primal import linear_svm_primal
from svm_dual import linear_svm_dual
from dataset import load_split_dataset
from evaluate import calculate_metrics_and_plot
import matplotlib.pyplot as plt
solvers.options['show_progress'] = False


def compute_kernel(kernel_type, X, X_prime, params):
    if  kernel_type == "rbf":
        sigma = params.get("sigma", None)
        if sigma is None:
            sigma = 1.0
            prtin("Compute with default value of sigma=1.0")
        dst = np.sqrt(np.sum((X[:, None, :] - X_prime[None, :, :]) ** 2, axis=2))
        kernel = np.exp(-dst ** 2 / (2 * sigma ** 2))

    elif kernel_type == "polynomial":
        degree = params.get("degree", None)
        if degree is None:
            degree = 3
            prtin("Compute with default value of degree=3")
        kernel = (np.sum(X[:, None]* X_prime, axis=2)+1)**degree
        
    else:
        raise Exception("invalid kernel type")

    return kernel


def kernel_svm(X, y, C, kernel_type, params):
    m, n = X.shape

    # define values of P, q, G, h, A, b
    kernel = compute_kernel(kernel_type, X, X, params)
    P = 0.5*np.outer(y, y) * kernel

    q = -np.ones(m)

    G = np.block([[-np.identity(m)],
                  [np.identity(m)]])

    h = np.block([np.zeros(m),
                  C*np.ones(m)])

    A = np.reshape(y, (1, m))

    b = 0.0

    # solve problem using cvxopt.solvers.qp
    sol = solvers.qp(matrix(P, tc='d'), matrix(q, tc='d'), matrix(G, tc='d'),
     matrix(h, tc='d'), matrix(A, (1, m), tc='d'), matrix(b, tc='d'))

    # extract Lagrange multipliers
    alpha = np.ravel(sol['x'])

    # find support vectors
    support_vectors = np.where((alpha > 1e-4))[0]

    # compute bias
    bias =  np.sum(-y[support_vectors] + np.sum(0.5*alpha[support_vectors] 
                                        * y[support_vectors]
                                        * kernel[:, support_vectors][support_vectors], 
                                        axis=0))
    bias /= (len(support_vectors))

    return alpha, support_vectors, bias


def predict(alpha, X, X_prime, y, bias, kernel_type, params):
    
    kernel = compute_kernel(kernel_type, X, X_prime, params)
    
    # predict scores and predict classes of data points
    y_pred_scores = np.sum(0.5*alpha * y * kernel.T, axis=1) - bias
    y_pred = np.sign(y_pred_scores)

    return y_pred, y_pred_scores


def plot_results(X, y):
    plt.figure(figsize=(8, 6))
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', s=50)
    
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('SVM')
    plt.show()


def plot_results_2d(alpha, X, y, bias, kernel_type, params):
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
    best_pred, _ = predict(alpha, X, Z, y, bias, kernel_type, params)
    Z = best_pred.reshape(x0.shape)
    plt.contourf(x0, x1, Z, cmap="rainbow", alpha=0.5)

    # plot training points
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap="viridis")

    # set labels
    plt.title('Kernel SVM')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.show()


def load_2d_datasets():
    # dataset = "Datasets/with slack"
    # dataset = "Datasets/without slack"
    # dataset = "Datasets/not linearly separable"
    dataset = "Datasets/moons"
    dataset = np.load("%s.npy" % dataset)
    X = dataset[:, :2]
    y = dataset[:, 2]
    y[y==0] = -1
    # shuffle dataset
    indices = np.random.randint(low=0, high=X.shape[0], size=X.shape[0])
    X = X[indices]
    y = y[indices]
    split = int(X.shape[0] * 0.8)
    X_train, X_test, y_train, y_test = X[:split, :], X[split:, :], y[:split], y[split:]

    return X_train, X_test, y_train, y_test


def svm():

    # load iris dataset
    # X_train, X_test, y_train, y_test = load_split_dataset()
    # ## convert label = 0 to label = -1
    # y_train[y_train==0] = -1
    # y_test[y_test==0] = -1

    # load 2d datasets
    X_train, X_test, y_train, y_test = load_2d_datasets()

    C = 1
    kernel_type, params = "rbf", {"sigma":1.0}
    alpha, support_vectors, bias = kernel_svm(X_train, y_train, C, kernel_type, params)
    y_pred, _ = predict(alpha, X_train, X_test, y_train, bias, kernel_type, params)
    
    # kernel_type, params = "polynomial", {"degree":3.0}
    # alpha, support_vectors, bias = kernel_svm(X_train, y_train, C, kernel_type, params)
    # y_pred, _ = predict(alpha, X_train, X_test, y_train, bias, kernel_type, params)
    
    print("Evaluation : ")
    calculate_metrics_and_plot(y_test, y_pred, labels=np.unique(y_train))
    y_pred[y_pred==-1] = 0
    plot_results_2d(alpha, X_train, y_train, bias, kernel_type, params)


if __name__== "__main__":
    svm()
    
