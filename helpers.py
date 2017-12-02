#
#

from __future__ import division
from scipy import sparse as sp
from scipy.sparse.linalg import spsolve

import numpy as np


class MathUtils:

    @staticmethod
    def als(y, lam, p, niter=10):
        """
            Asymmetric Least Squares Smoothing

            :param y:

            :param lam:
                cost of non-smoothness

            :param p:
                assymetry coefficient: +1.0 is smoothing on peaks, -1.0 is smoothing on dips

            :param niter:
                max iterations count

            :return:
        """
        L = len(y)
        D = sp.csc_matrix(np.diff(np.eye(L), 2))
        w = np.ones(L)
        for i in range(niter):
            W = sp.spdiags(w, 0, L, L)
            Z = W + lam * D.dot(D.transpose())
            z = spsolve(Z, w*y)
            w = p * (y > z) + (1-p) * (y < z)
        return z
