# -*- coding: utf-8 -*-
""" GPML Likelihood functions.

Observation model for data described by a Gaussian process.

Created: Fri Jan 18 10:29:06 2013 by Hannes Nickisch, Philips Research Hamburg.
Modified: $Id: covariances.py 1263 2013-12-13 13:36:13Z hn $
"""
__version__ = "$Id: likelihood.py 913 2013-08-15 12:54:33Z hn $"

import numpy as np
import scipy.special as ssp
from hyper import HyperIter

def logphi(z):
    """ Safe implementation of the log of phi(x) = \int_{-\infty}^x N(f|0,1) df
        logphi(z) = log(normcdf(z)), normcdf(z) = (1+erf(z/sqrt(2)))/2
    """
    shp = np.shape(z)
    z = np.asarray(z).reshape(-1,1)
    lp = np.zeros_like(z)                                     # allocate memory
    zmin,zmax = -6.2,-5.5
    ok = z>zmax                              # safe evaluation for large values
    bd = z<zmin                                               # use asymptotics
    ip = ~ok & ~bd                           # interpolate between both of them
    lam = 1/(1+np.exp( 25*(0.5-(z[ip]-zmin)/(zmax-zmin)) ))   # interp. weights
    lp[ok] = np.log( (1+ssp.erf(z[ok]/np.sqrt(2)))/2 )
    #  use lower and upper bound acoording to Abramowitz&Stegun 7.1.13 for z<0
    # lower -log(pi)/2 -z.^2/2 -log( sqrt(z.^2/2+2   ) -z/sqrt(2) )
    # upper -log(pi)/2 -z.^2/2 -log( sqrt(z.^2/2+4/pi) -z/sqrt(2) )
    # the lower bound captures the asymptotics
    lp[~ok] = -np.log(np.pi)/2 -z[~ok]**2/2 -np.log( np.sqrt(z[~ok]**2/2+2)
                                                           -z[~ok]/np.sqrt(2) )
    lp[ip] = (1-lam)*lp[ip] + lam*np.log( (1+ssp.erf(z[ip]/np.sqrt(2)))/2 )
    if shp==(): return np.float(lp)
    else:       return lp.reshape(shp)

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
        elif mode=='pred':
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
        elif mode=='pred':
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
    def __init__(self):
        """ Construct an error function likelihood.
        """
        Likelihood.__init__(self,hyp={},name='erf')
    
    def cum_gauss(self,f,y=None):
        if y==None: yf = f                      # product of latents and labels
        else:       yf = y*f
        p = (1+ssp.erf(yf/np.sqrt(2)))/2
        return p,logphi(yf)

    def gau_over_cum_gauss(self,f,p):
        shp = np.shape(f)
        f,p = np.asarray(f).reshape(-1,1),np.asarray(p).reshape(-1,1)
        n_p = np.zeros_like(f)                                # allocate memory
        ok = f>-5                      # naive evaluation for large values of f
        n_p[ok] = (np.exp(-f[ok]**2/2)/np.sqrt(2*np.pi)) / p[ok]
        bd = f<-6                                # tight upper bound evaluation
        n_p[bd] = np.sqrt(f[bd]**2/4+1)-f[bd]/2
        ip = ~ok & ~bd              # linearly interpolate between both of them
        t,lam = f[ip],-5-f[ip]
        n_p[ip] = (1-lam)*(np.exp(-t**2/2)/np.sqrt(2*np.pi))/p[ip] + lam*(
                                                         np.sqrt(t**2/4+1)-t/2)
        if shp==(): return np.float(n_p)
        else:       return n_p.reshape(shp)

    def evaluate(self,y,fmu,fs2=0,mode='lp',i=None):
        p,lp = self.cum_gauss(fmu,y)
        if   mode=='lp':
            if not np.isscalar(fs2) or fs2!=0:
                p,lp = self.cum_gauss(fmu/np.sqrt(1+fs2),y)
            return lp
        elif   mode=='pred':
            return 2*p-1,4*p*(1-p)
        elif mode=='LA':
            n_p = self.gau_over_cum_gauss(y*fmu,p)
            dlp  = y*n_p                                 # 1st deriv of log lik
            d2lp = -n_p**2 - y*fmu*n_p                   # 2nd deriv of log lik
            d3lp = 2*y*n_p**3 +3*fmu*n_p**2 +y*(fmu**2-1)*n_p # 3rd deriv of ll
            return dlp,d2lp,d3lp
        elif mode=='EP':
            z = fmu/np.sqrt(1+fs2)
            _,lZ = self.cum_gauss(z,y)
            n_p = self.gau_over_cum_gauss(y*z,np.exp(lZ))
            dlZ = y*n_p/np.sqrt(1+fs2)                # 1st derivative wrt mean
            d2lZ = -n_p*(y*z+n_p)/(1+fs2)                      # 2nd derivative
            return dlZ,d2lZ
        elif mode=='VB':
            return
#            d =  0.158482605320942
#            n = numel(s2); b = d*y.*ones(n,1); z = zeros(n,1);
#            varargout = {b,z};
        else:
            print mode
            raise Exception('Unknown mode.')