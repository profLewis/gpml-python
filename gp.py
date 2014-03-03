# -*- coding: utf-8 -*-
""" GPML main function.

Perform Gaussian process inference and prediction.

Created: Fri Jan 18 10:29:06 2013 by Hannes Nickisch, Philips Research Hamburg.
Modified: $Id: gp.py 1263 2013-12-13 13:36:13Z hn $
"""
__version__ = "$Id: gp.py 913 2013-08-15 12:54:33Z hn $"

import numpy as np
from covariance import se
from mean import zero,one
from likelihood import gauss
from inference import exact

# Compatibility gp.inf.func_name <=> gp.lik.name
comp = {'exact':set('gauss'),'ep':set('gauss','erf')}

class GaussianProcess:
    """ Main object to perform Gaussian process calculations.
    """
    def __init__(self,inf=exact,cov=1.0*se(ell=1.0),
                                  mean=zero(),lik=gauss(sn=1.0),X=None,y=None):
        """ Set up a gp, perform inference if X and y are provided.
        """
        self.inf = inf                                       # inference method
        self.cov,self.mean,self.lik = cov,mean,lik # covariance,mean,likelihood
        self.post = None                         # init posterior approximation
        if X!=None and y!=None: self.post = self.inference(X,y)  # do inference
    
    def inference(self,X,y,deriv=False):
        """ Perform approximate inference.
        """
        out = self.inf(self.cov,self.mean,self.lik,X,y,deriv=deriv)
        self.post = out[0]                      # store posterior approximation
        return out

    def predict(self,X,y,Xs,ys=0):
        """ Perform prediction.
        """
        if self.post==None: raise Exception('Perform inference first!')
        nb,ns,na = 1000,Xs.shape[0],0          # batch size for loop processing
        fmu,fs2 = np.zeros(ns),np.zeros(ns)
        ymu,ys2 = np.zeros(ns),np.zeros(ns)
        lp = np.zeros(ns)
        while na<ns:
            idx = np.arange(na,min(na+nb,ns))
            kss = self.cov(Xs[idx],diag=True)                   # self variance
            Ks = self.cov(Xs[idx],X)                        # cross-covariances
            ms = self.mean(Xs[idx])
            
            al,sW,L,C = self.post.alpha,self.post.sW,self.post.L,self.post.C
            fmu[idx] = ms + np.dot(Ks,al)
            if L==None:
                fs2[idx] = kss + np.sum(Ks*np.dot(Ks,L),axis=1)
            else:
                V = np.linalg.solve(L,sW*Ks.T)
                fs2[idx] = kss - np.sum(V*V,axis=0)
            if ys==0: yi = 0
            else:     yi = ys[idx]
            lp[idx],ymu[idx],ys2[idx] = self.lik.pred(yi,fmu[idx],fs2[idx])
            na += nb

        return fmu,fs2,ymu,ys2,lp

if __name__ == "__main__":
    def f(x): return x * np.sin(x)
    X = np.atleast_2d([1., 3., 5., 6., 7., 8.]).T
    y = f(X).ravel()
    Xs = np.atleast_2d(np.linspace(0, 10, 2007)).T
    from gp import GaussianProcess as gp
    gp = gp(mean=1.0*one())
    post,nlZ,dnlZ = gp.inference(X,y,deriv=True)
    fmu,fs2,ymu,ys2,lp = gp.predict(X,y,Xs)