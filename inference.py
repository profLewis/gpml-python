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

def ep(cov,mean,lik,X,y,deriv=False):
    """ Perform expectation propagation based approximate inference for a GP.
    """
    tol,min_sweep,max_sweep = 1e-4,2,10       # tolerance to stop EP iterations
    n = X.shape[0]
    K,m = cov(X),mean(X)                         # evaluate mean and covariance
    class post: C,L,sW,alpha = None,None,None,None            # allocate result

    def compute_parms(K,y,ttau,tnu,lik,m):
        return Sigma,mu,L,alpha,nlZ

    # A note on naming: variables are given short but descriptive names in 
    # accordance with Rasmussen & Williams "GPs for Machine Learning" (2006):
    # mu and s2 are mean and variance, nu and tau are natural parameters. A
    # leading t means tilde, a subscript _ni means "not i" (for cavity 
    # parameters), or _n for a vector of cavity parameters.
    # N(f|mu,Sigma) is the posterior.

    ttau,tnu = np.zeros([n,1]),np.zeros([n,1])                 # init with zero
    Sigma,mu = K,m              # initialize Sigma and mu, the parameters of ..
    nlZ = -np.sum(lik.pred(y,m,np.diag(K))[0])    # .. the Gaussian post approx
    
    nlZ_old,sweep = np.inf,0           # converged, max. sweeps or min. sweeps?
    alpha = 0
    L = 0

    if sweep==max_sweep and abs(nlZ-nlZ_old)>tol:
        raise Exception('maximum number of sweeps exceeded')
    while (abs(nlZ-nlZ_old) > tol and sweep < max_sweep) or sweep<min_sweep:
        nlZ_old = nlZ; sweep += 1
        # iterate EP updates (in random order) over examples
        for i in np.random.permutation(n):
            # first find the cavity distribution parameters
            tau_ni,nu_ni = 1/Sigma[i,i]-ttau[i],mu[i]/Sigma[i,i]-tnu[i]

            # compute derivatives of the indivdual log partition function
            lZ,dlZ,d2lZ = lik.ep(y[i],nu_ni/tau_ni, 1/tau_ni)
            ttau_old,tnu_old = ttau[i],tnu[i] # find new tilde params, keep old
            ttau[i] =                     -d2lZ  /(1+d2lZ/tau_ni)
            ttau[i] = max(ttau[i],0) # enforce >0 i.e. lower bound ttau by zero
            tnu[i]  = ( dlZ - nu_ni/tau_ni*d2lZ )/(1+d2lZ/tau_ni)

            dtt,dtn = ttau[i]-ttau_old,tnu[i]-tnu_old  # rank-1 update Sigma ..
            si = Sigma[:,i]; ci = dtt/(1+dtt*si[i])
            Sigma -= ci*np.outer(si,si)               # takes 70% of total time
            mu -= (ci*(mu[i]+si[i]*dtn)-dtn)*si           # .. and recompute mu

        # recompute since repeated rank-one updates destroy numerical precision
#        Sigma,mu,L,alpha,nlZ = compute_parms(K,y,ttau,tnu,lik,m)

    post.alpha,post.sW,post.L = alpha,np.sqrt(ttau),L # return posterior params

    if not deriv: return post,nlZ                     # no derivatives required
    else:
        class dnlZ: cov,mean,lik = None,None,None             # allocate memory
        
        
        return post,nlZ,dnlZ

def laplace(cov,mean,lik,X,y,deriv=False):
    """ Perform Laplace approximation based approximate inference for a GP.
    """
    n = X.shape[0]
    K,m = cov(X),mean(X)                         # evaluate mean and covariance
    class post: C,L,sW,alpha = None,None,None,None            # allocate result

    nlZ = 0
    if not deriv: return post,nlZ                     # no derivatives required
    else:
        class dnlZ: cov,mean,lik = None,None,None             # allocate memory
        return post,nlZ,dnlZ