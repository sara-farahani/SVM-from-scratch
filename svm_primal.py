import numpy as np
from cvxopt import matrix, solvers


def linear_svm_primal(X, y, C=1):
    m, n = X.shape
    
    # define values of P, q, G, h, A, b
    P = np.block([[2*np.identity(n), np.zeros((n, m+1))], 
                  [np.zeros((m+1, n)), np.zeros((m+1, m+1))]])

    q = np.block([np.zeros(n+1), C*np.ones(m)])

    G = np.block([[-X*np.reshape(y, (-1, 1)), y.reshape(-1,1), -np.identity(m)],
                  [np.zeros((m, n+1)), -np.identity(m)]])

    h = np.block([-np.ones(m), np.zeros(m)])
    
    # solve problem using cvxopt.solvers.qp
    sol = solvers.qp(matrix(P, tc='d'), matrix(q, tc='d'), matrix(G, tc='d'),
                     matrix(h, tc='d'))

    # extract weights, bias and distances 
    weights = np.array(sol['x'][0:n])
    bias = float(sol['x'][n])
    distances = np.array(sol['x'][n+1:])

    return weights, bias, distances