# -*- coding: utf-8 -*-
""" GPML regression example.

Pairwise dependencies between datapoints to be used in a Gaussian process.

Created: Fri Jan 18 10:29:06 2013 by Hannes Nickisch, Philips Research Hamburg.
Modified: $Id: example.py 1263 2013-12-13 13:36:13Z hn $
"""
__version__ = "$Id: example.py 913 2013-08-15 12:54:33Z hn $"

import numpy as np
import matplotlib.pyplot as plt

def show_1d(X,y,Xs,ym,ys):
    plt.fill(np.concatenate([Xs, Xs[::-1]]),
             np.concatenate([ym - 1.96*ys,(ym + 1.96*ys)[::-1]]),
             alpha=0.25, fc='k', ec='k', label='95% confidence interval')
    plt.plot(X,y,'b+',ms=10)
    plt.plot(Xs,ym,'b',lw=2)
    plt.grid()
    plt.xlabel('input X')
    plt.ylabel('output y')

def f(x): return x * np.sin(x)
X = np.atleast_2d([1., 3., 5., 6., 7., 8.]).T
y = f(X).ravel()
Xs = np.atleast_2d(np.linspace(0, 10, 2007)).T

from sklearn import gaussian_process
gp = gaussian_process.GaussianProcess(corr='cubic', theta0=1e-2, thetaL=1e-4, thetaU=1e-1, random_start=100)
gp.fit(X,y)
ymn,ys2 = gp.predict(Xs, eval_MSE=True); ysd = np.sqrt(ys2)

from gp import GaussianProcess as gp
gp = gp(X=X,y=y)
post = gp.inference(X,y)
fmu,fs2,ymu,ys2,lp = gp.predict(X,y,Xs)


f0 = plt.figure(0); plt.clf()
show_1d(X,y,Xs,ymn,ysd)
f1 = plt.figure(1); plt.clf()
show_1d(X,y,Xs,ymu,ys2)