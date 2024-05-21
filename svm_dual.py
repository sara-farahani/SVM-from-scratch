import numpy as np
from cvxopt import matrix, solvers


def linear_svm_dual(X, y, C=1):
    m, n = X.shape

     # define values of P, q, G, h, A, b
    linear_kernel = np.dot(X, X.T)
    P = 0.5*np.outer(y, y) * linear_kernel

    q = -np.ones(m)

    G = np.block([[-np.identity(m)],
                  [np.identity(m)]])

    h = np.block([-np.zeros(m),
                  C*np.ones(m)])

    A = np.reshape(y, (1, m))

    b = 0.0

    # solve problem using cvxopt.solvers.qp
    sol = solvers.qp(matrix(P, tc='d'), matrix(q, tc='d'), matrix(G, tc='d'),
                    matrix(h, tc='d'), matrix(A, (1, m), tc='d'), matrix(b, tc='d'))

    # extract Lagrange multipliers
    alphas = np.ravel(sol['x'])

    return alphas