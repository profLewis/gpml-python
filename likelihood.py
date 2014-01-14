# -*- coding: utf-8 -*-
""" GPML Likelihood functions.

Observation model for data described by a Gaussian process.

Created: Fri Jan 18 10:29:06 2013 by Hannes Nickisch, Philips Research Hamburg.
Modified: $Id: covariances.py 1263 2013-12-13 13:36:13Z hn $
"""
__version__ = "$Id: likelihood.py 913 2013-08-15 12:54:33Z hn $"

import numpy as np
from hyper import HyperIter

class Likelihood(HyperIter):
    """ Likelihood function base class.
    """
    def __init__(self,hyp={},name='plain'):
        self.hyp = hyp
        self.name = name
        self.chil = []
    
    def pred(self,y,fmu,fs2=0):
        """ Prediction mode.
            Evaluate the predictive distribution. Let p(y_*|f_*) be the
            likelihood of a test point and N(f_*|fmu,fs2) an approximation to
            the posterior marginal p(f_*|x_*,x,y) as returned by an inference
            method. The predictive distribution p(y_*|x_*,x,y) is approximated
            by:   q(y_*) = \int N(f_*|fmu,fs2) p(y_*|f_*) df_*
                lp = log( q(y) ) for a particular value of y,
                    if fs2 is None or 0, this corresponds to log( p(y|fmu) ).
                ymu,ys2 are mean and variance of the predictive marginal q(y)
                    Note that these values do not depend on y.
        """
        lp = self.evaluate(y,fmu,fs2,mode='lp')
        ymu,ys2 = self.evaluate(y,fmu,fs2,mode='pred')
        return lp,ymu,ys2
    
    def la(self,y,f,i=None):
        """ Laplace approximation mode.
            Evaluate derivatives of log(p(y|f)) w.r.t. the latent location f.
                  lp = log( p(y|f) )
                 dlp = d   log( p(y|f) ) / df
                d2lp = d^2 log( p(y|f) ) / df^2
                d3lp = d^3 log( p(y|f) ) / df^3
            If di has a value different from 0 and None, we return derivatives
            w.r.t. the ith hyperparameter.
                  lp_dh = d   log( p(y|f) ) / (     dhyp_i)
                 dlp_dh = d^2 log( p(y|f) ) / (df   dhyp_i)
                d2lp_dh = d^3 log( p(y|f) ) / (df^2 dhyp_i)
        """
        try:    self.evaluate(1.0,0.0,mode='LA')
        except: raise NotImplementedError("%s/LA not implemented."%self.name)
        if i==None:                                       # for plain inference
            lp = self.evaluate(y,f,mode='lp')
            dlp,d2lp,d3lp = self.evaluate(y,f,mode='LA')
            return lp,dlp,d2lp,d3lp
        else:                                              # return derivatives
            if self.get_nhyp()==0:
                return None,None,None
            else:
                lp_dh,dlp_dh,d2lp_dh = self.evaluate(y,f,mode='LA',i=i)
                return lp_dh,dlp_dh,d2lp_dh

    def ep(self,y,fmu,fs2,i=None):
        """ Expectation propagation mode.
            Evaluate derivatives of the partition function Z given by
            Z = \int p(y|f) N(f|fmu,fs2) df.
                  lZ =     log(Z)
                 dlZ = d   log(Z) / dfmu
                d2lZ = d^2 log(Z) / dfmu^2
            If di has a value different from 0 and None, we return derivatives
            w.r.t. the ith hyperparameter.
                lZ_dh = d log(Z) / dhyp_i
        """
        try:    self.evaluate(1.0,0.0,fs2=1.0,mode='EP')
        except: raise NotImplementedError("%s/EP not implemented."%self.name)
        if i==None:                                       # for plain inference
            lZ = self.evaluate(y,fmu,fs2=fs2,mode='lp')
            dlZ,d2lZ = self.evaluate(y,fmu,fs2=fs2,mode='EP')
            return lZ,dlZ,d2lZ
        else:                                              # return derivatives
            if self.get_nhyp()==0:
                return None,None,None
            else:
                lZ_dh = self.evaluate(y,fmu,fs2=fs2,mode='EP',i=i)
                return lZ_dh

    def vb(self,y):
        """ Variational Bayes mode.
            ga is the variance of a Gaussian lower bound to the likelihood.
            p(y|f) \ge exp( b*(f+z) - (f+z).^2/(2*ga) - h(ga)/2 )
                   \propto N(f|b*ga-z,ga)
            The function returns the linear part b and z.
        """
        try:    self.evaluate(1.0,0.0,mode='VB')
        except: raise NotImplementedError("%s/VB not implemented."%self.name)
        b,z = self.evaluate(y,None,mode='VB')
        return b,z

    def evaluate(self,y,fmu,fs2=0,mode='lp',i=None):
        """ Default evaluation function to be overridden by derived classes.
        """
        if   mode=='lp':
            raise NotImplementedError
        if   mode=='pred':
            raise NotImplementedError
        elif mode=='LA':
            raise NotImplementedError
        elif mode=='EP':
            raise NotImplementedError
        elif mode=='VB':
            raise NotImplementedError
        else: raise Exception('Unknown mode.')

class gauss(Likelihood):
    def __init__(self,sn=None,log_sn=None):
        """ Construct a Gaussian likelihood.
        """
        if log_sn==None: log_sn = np.log(sn)
        hyp = dict(log_sn=log_sn)
        Likelihood.__init__(self,hyp=hyp,name='gauss')
    
    def evaluate(self,y,fmu,fs2=0,mode='lp',i=None):
        sn2 = np.exp(2*self.hyp['log_sn'])              # obtain hyperparameter
        ymmu = y-fmu
        if   mode=='lp':
            return -ymmu**2/(sn2+fs2)/2 - np.log(2*np.pi*(sn2+fs2))/2
        if   mode=='pred':
            return fmu,fs2+sn2
        elif mode=='LA':
            if i==None:
                dlp  = ymmu/sn2;                    # 1st derivative of log lik
                d2lp = -np.ones_like(ymmu)/sn2      # 2nd derivative of log lik
                d3lp = np.zeros_like(ymmu)          # 3rd derivative of log lik
                return dlp,d2lp,d3lp
            else:
                lp_dh = ymmu**2/sn2 - 1   # derivative of log lik w.r.t. hypers
                dlp_dh = -2*ymmu/sn2               # first derivative, and also
                d2lp_dh = 2*np.ones_like(ymmu)/sn2       # second mu derivative
                return lp_dh,dlp_dh,d2lp_dh
        elif mode=='EP':
            if i==None:
                dlZ = ymmu/(sn2+fs2)                  # 1st derivative of log Z
                d2lZ = -1/(sn2+fs2)                   # 2nd derivative of log Z
                return dlZ,d2lZ
            else:
                lZ_dh = ymmu**2/(sn2+fs2) - 1  # deriv of log lik w.r.t. hypers
                return lZ_dh/(1+fs2/sn2)
        elif mode=='VB':
            return 0,y
        else: raise Exception('Unknown mode.')

class erf(Likelihood):
    def __init__(self,sn=None,log_sn=None):
        """ Construct an error function likelihood.
        """
        Likelihood.__init__(self,hyp={},name='erf')

    def evaluate(self,y,fmu,fs2=0,mode='lp',i=None):
        if   mode=='lp':
            raise NotImplementedError
        if   mode=='pred':
            raise NotImplementedError
        elif mode=='LA':
            raise NotImplementedError
        elif mode=='EP':
            raise NotImplementedError
        elif mode=='VB':
            raise NotImplementedError
        else: raise Exception('Unknown mode.')