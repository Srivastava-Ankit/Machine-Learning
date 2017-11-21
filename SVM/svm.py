import numpy as np
import gzip
import pickle
from collections import Counter, defaultdict
from numpy import median
import argparse
from sklearn import svm

kINSP = np.array([(1, 8, +1),
               (7, 2, -1),
               (6, -1, -1),
               (-5, 0, +1),
               (-5, 1, -1),
               (-5, 2, +1),
               (6, 3, +1),
               (6, 1, -1),
               (5, 2, -1)])

kSEP = np.array([(-2, 2, +1),    # 0 - A
              (0, 4, +1),     # 1 - B
              (2, 1, +1),     # 2 - C
              (-2, -3, -1),   # 3 - D
              (0, -1, -1),    # 4 - E
              (2, -3, -1),    # 5 - F
              ])


def weight_vector(x, y, alpha):
    """
    Given a vector of alphas, compute the primal weight vector w.
    The vector w should be returned as an Numpy array.
    """

    w = np.zeros(len(x[0]))
    # TODO: IMPLEMENT THIS FUNCTION

    for i in range(len(x)):
        w = w + alpha[i]* x[i] * y[i]
    return w



def find_support(x, y, w, b, tolerance=0.001):
    """
    Given a set of training examples and primal weights, return the indices
    of all of the support vectors as a set.
    """

    support = set()
    # TODO: IMPLEMENT THIS FUNCTION
    for i in range(len(x)):
         c = (sum(w * x[i]) + b) * y[i]
         if (c < 1 + tolerance and c > 1 - tolerance):
            support.add(i)
    return support



def find_slack(x, y, w, b):
    """
    Given a set of training examples and primal weights, return the indices
    of all examples with nonzero slack as a set.
    """

    slack = set()
    # TODO: IMPLEMENT THIS FUNCTION
    for i in range(len(x)):
         c = (sum(w * x[i]) + b) * y[i]
         if c < 1:
             slack.add(i)
    return slack






