# -*- coding: utf-8 -*-
""" GPML Inference methods.

Approximate inference methods.

Created: Fri Jan 18 10:29:06 2013 by Hannes Nickisch, Philips Research Hamburg.
Modified: $Id: inference.py 1263 2013-12-13 13:36:13Z hn $
"""
__version__ = "$Id: inference.py 913 2013-08-15 12:54:33Z hn $"

import numpy as np

def chol(A):
    """ Cholesky factor of A so that A = L*L' = np.dot(L,L.T).
    """
    return np.linalg.cholesky(A)

def solve_chol(L,B):
    """ Solve linear equation from Cholesky factorisation so that
        np.linalg.solve(A,B) = solve_chol(chol(A),B).
    """
    return np.linalg.solve(L.T,np.linalg.solve(L,B))

def exact(cov,mean,lik,X,y,deriv=False):
    """ Perform exact inference for a GP with Gaussian likelihood.
    """
    n = X.shape[0]
    K,m = cov(X),mean(X)                         # evaluate mean and covariance
    sn2 = np.exp(2*lik.hyp['log_sn'])
    class post: C,L,sW,alpha = None,None,None,None            # allocate result

    if sn2<1e-4:                      # very tiny sn2 lead to numerical trouble
        L,sl = chol(K+sn2*np.eye(n)),1    # Cholesky factor of noisy covariance
        post.L = -solve_chol(L,np.eye(n));              # L = -inv(K+inv(sW^2))
    else:
        L,sl = chol(K/sn2+np.eye(n)),sn2                 # Cholesky factor of B
        post.L = L                                 # L = chol(eye(n)+sW*sW'.*K)
    al = solve_chol(L,y-m)/sl
    post.alpha = al                                      # posterior parameters
    post.sW = np.ones([n,1])/np.sqrt(sn2)      # sqrt of noise precision vector

    nlZ = np.dot(y-m,al/2) + np.sum(np.log(np.diag(L)))               # compute
    nlZ += n*np.log(2*np.pi*sl)/2            # negative log marginal likelihood
    if not deriv: return post,nlZ                     # no derivatives required
    else:
        class dnlZ: cov,mean,lik = None,None,None             # allocate memory
        Q = solve_chol(L,np.eye(n))/sl - np.outer(al,al)           # precompute
        if len(cov.get_hyp())>0: dnlZ.cov = np.zeros(len(cov.get_hyp()))
        for i in range(len(cov.get_hyp())):
            dnlZ.cov[i] = np.sum(cov(X,deriv=i)*Q)/2
        dnlZ.lik = sn2*np.trace(Q)
        if len(mean.get_hyp())>0: dnlZ.mean = np.zeros(len(mean.get_hyp()))
        for i in range(len(mean.get_hyp())):
            dnlZ.mean[i] = -np.dot(al,mean(X,deriv=i))
        return post,nlZ,dnlZ