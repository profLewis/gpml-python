# -*- coding: utf-8 -*-
""" GPML Covariance functions.

Pairwise dependencies between datapoints to be used in a Gaussian process.

Created: Wed Dec 18 16:57:01 2013 by Hannes Nickisch, Philips Research Hamburg.
Modified: $Id: covariance.py 1263 2013-12-13 13:36:13Z hn $
"""
__version__ = "$Id: covariance.py 913 2013-08-15 12:54:33Z hn $"

# TODO: mask

import numpy as np
import hashlib as hl
from functools import partial
import scipy.spatial.distance as ssd
from hyper import HyperIter

class Covariance(HyperIter):
    """ Covariance function base class.
    """
    def __init__(self,hyp={},name='plain',chil=[]):
        self.hyp = hyp
        self.name,self.chil = name,chil

    def __str__(self):
        return "cov/"+self.name

    def __repr__(self):
        return self.__str__()

    def __mul__(self, other):
        """ Product of a covariance function and a scalar or another
            covariance function.
        """
        if isinstance(other,float) or isinstance(other,int):
            hyp = dict(log_sf=0.5*np.log(other))
            other = Covariance(hyp=hyp,name='const')
        if self.name=='prod':    # make sure all terms appear on the same level
            self.chil.append(other)
            return self
        else:
            return Covariance(name='prod',chil=[other,self])
    def __rmul__(self, other):                # other is not of type Covariance
        return self.__mul__(other)

    def __add__(self, other):
        """ Sum of a covariance function and a scalar or another
            covariance function.
        """
        if isinstance(other,float) or isinstance(other,int):
            hyp = dict(log_sf=0.5*np.log(other))
            other = Covariance(hyp=hyp,name='const')
        if self.name=='sum':    # make sure all terms appear on the same level
            self.chil.append(other)
            return self
        else:
            return Covariance(name='sum',chil=[other,self])
    def __radd__(self, other):                # other is not of type Covariance
        return self.__add__(other)

    def evaluate(self,x,z=None, diag=False,deriv=None):
        """ Default evaluation function to be overridden by derived classes.
        """
        def shape(x,z,diag):                    # shape of the resulting matrix
            n = x.shape[0]
            if diag:        return n,
            else:
                if z==None: return n,n
                else:       return n,z.shape[0]
        if self.name in ['prod','sum']:
            if deriv==None:                                          # evaluate
                if   self.name=='prod':
                    K = np.ones( shape(x,z,diag))
                    for k in self.chil: K *= k(x,z=z,diag=diag)
                elif self.name=='sum' :
                    K = np.zeros(shape(x,z,diag))
                    for k in self.chil: K += k(x,z=z,diag=diag)
            else:                                        # evaluate derivatives
                deriv = int(deriv)                           # derivative index
                if deriv<0 or deriv>self.get_nhyp()-1:
                    raise Exception('Bad derivative index.')
                nhyp = np.cumsum([c.get_nhyp() for c in self.chil])
                j = np.nonzero(deriv<nhyp)[0].min()             # index of term
                if j==0: i = deriv                          # index inside term
                else:    i = deriv-nhyp[j-1]
                if   self.name=='prod':
                    K = np.ones( shape(x,z,diag))
                    for l,k in enumerate(self.chil):
                        if l==j: K *= k(x,z=z,diag=diag,deriv=i)
                        else:    K *= k(x,z=z,diag=diag)
                elif self.name=='sum' :
                    K = self.chil[j](x,z=z,diag=diag,deriv=i)
        elif self.name=='const':
            K = np.exp(2*self.hyp['log_sf'])*np.ones(shape(x,z,diag))
            if deriv!=None: K *= 2.0
        else:                                                    # unknown name
            K = np.zeros(shape(x,z,diag))
        return K

    def __call__(self, x,z=None, diag=False,deriv=None):
        return self.evaluate(x,z=z,diag=diag,deriv=deriv)

def memoise(f=None,n=10,output=False):
    """ Add a cache to an existing function depending on numpy arrays.
    """
    if f is None: return partial(memoise,n=n,output=output)

    def hashfun(x):
        """ Hash function also dealing with numpy arrays.
            Output is always a string.
        """
        if type(np.array([]))==type(x): return hl.sha1(x).hexdigest()
        else: return str(hash(x))

    cache = {}
    def g(*args,**kwargs):
        """ Augmented function as returned by the decorator.
        """
        for c in cache:                 # make all cache entries one tick older
            v,a = cache[c]; cache[c] = v,a+1
        h = ''                                             # overall hash value
        for a in args:            h += hashfun(a)
        for v in kwargs.values(): h += hashfun(v)
        x = hashfun(h)
        if x not in cache:                                         # cache miss
            if output: print "miss(%d)"%len(cache)
            if len(cache)>=n:            # cache is full -> delete oldest entry
                if output: print "delete"
                xold,age = None,0
                for xact in cache:
                    if cache[xact][1]>age: xold,age = xact,cache[xact][1]
                del cache[xold]
            cache[x] = f(*args,**kwargs),0
        else:                                            # cache hit, reset age
            if output: print "hit(%d)"%len(cache)
            cache[x] = cache[x][0],0
        return cache[x][0]

    return g

@memoise(n=5)
def sq_dist_scipy(x,z):
    """ Compute squared distances using scipy.spatial.distance.
    """
    return ssd.cdist(x,z,'sqeuclidean')

@memoise(n=5)
def sq_dist_dot(x,z):
    """ Compute squared distances by dot products.
    """
    d2 = np.sum(x*x,axis=1).reshape(-1,1) + np.sum(z*z,axis=1).reshape(1,-1)
    d2 -= 2*np.dot(x,z.T)
    return d2

@memoise(n=5)
def sq_dist_loop(x,z):
    """ Compute squared distances by a loop over dimensions.
    """
    d2 = np.zeros([x.shape[0],z.shape[0]])
    for i in range(x.shape[1]):
        d2 += (x[:,i].reshape(-1,1)-z[:,i].reshape(1,-1))**2
    return d2

def sq_dist(x,z=None,ell=None,sqd=sq_dist_scipy):
    """ Compute a matrix of all pairwise squared distances
        between two sets of vectors, stored in the row of the two matrices:
        x (of size n by D) and z (of size m by D).
    """
    if ell==None:
        if z==None: return sqd(x,x)
        else:       return sqd(x,z)
    else:
        ell = ell.reshape(1,-1)
        if z==None: return sqd(x/ell,x/ell)
        else:       return sqd(x/ell,z/ell)

class noise(Covariance):
    def __init__(self):
        """ Construct a noise covariance function.
        """
        Covariance.__init__(self,hyp={},name='noise')

    def evaluate(self, x,z=None, diag=False,deriv=None):
        """ Evaluation of noise covariance function.
        """
        n = x.shape[0]
        if z==None: m = n
        else:       m = z.shape[0]
        if diag:
            if deriv==None: K = np.ones(n)
            else:           K = np.zeros(n)
        else:
            if deriv==None: K = np.eye(n,m)
            else:           K = np.zeros([n,m])
        return K

class stat(Covariance):
    def __init__(self,h=None,dh=None,log_ell=None):
        """ Construct a generic stationary covariance function
            k(x,z) = h(d2(x,z)), where d2(x,z) is the squared distance between
            the data points x and z.
        """
        hyp = dict(log_ell=log_ell)
        Covariance.__init__(self,hyp=hyp,name='stat')
        self.h,self.dh = h,dh

    def evaluate(self, x,z=None, diag=False,deriv=None):
        """ Evaluation of a generic stationary covariance function.
        """
        n,D = x.shape
        ell = np.exp(self.hyp['log_ell'])
        iso = np.size(ell)==1    # do we have an isotropic covariance function?
        if not iso: ell = ell.reshape(1,D)
        if diag: K = np.zeros(n)
        else:    K = sq_dist(x,z,ell=ell)
        if deriv==None: K = self.h(K)                   # covariance evaluation
        else:                                               # ell derivative(s)
            if iso:
                K = -2*self.dh(K)*K
            else:
                K = -2*self.dh(K)
                if diag: K *= 0
                else:
                    i = deriv
                    xi = (x[:,i]/ell[:,i]).reshape(n,1)
                    if z==None: K *= sq_dist(xi,None)
                    else:       K *= sq_dist(xi,(z[:,i]/ell[:,i]).reshape(-1,1))
        return K            

class se(Covariance):
    def __init__(self,ell=None,log_ell=None):
        """ Construct a squared exponential covariance function.
        """
        if log_ell==None: log_ell = np.log(ell)
        hyp = dict(log_ell=log_ell)
        Covariance.__init__(self,hyp=hyp,name='se')

    def evaluate(self, x,z=None, diag=False,deriv=None):
        """ Evaluation of squared exponential covariance function.
        """
        def  h(D2): return      np.exp(-0.5*D2)
        def dh(D2): return -0.5*np.exp(-0.5*D2)
        k = stat(h=h,dh=dh,log_ell=self.hyp['log_ell'])
        return k.evaluate(x,z=z,diag=diag,deriv=deriv)

class rq(Covariance):
    def __init__(self,ell=None,log_ell=None,al=None,log_al=None):
        """ Construct a rational quadratic covariance function.
        """
        if log_ell==None: log_ell = np.log(ell)
        if log_al ==None: log_al  = np.log(al)
        hyp = dict(log_ell=log_ell,log_al=log_al)
        Covariance.__init__(self,hyp=hyp,name='rq')

    def evaluate(self, x,z=None, diag=False,deriv=None):
        """ Evaluation of rational quadratic covariance function.
        """
        al = np.exp(self.hyp['log_al'])
        def  g(D2): return 1+0.5*D2/al
        def  h(D2): return      g(D2)**(-al  )
        def dh(D2): return -0.5*g(D2)**(-al-1)
        k = stat(h=h,dh=dh,log_ell=self.hyp['log_ell'])
        if deriv==None:
            K = k.evaluate(x,z=z,diag=diag,deriv=deriv)
        else:
            if deriv==0:                          # log_al comes before log_ell
                ell = np.exp(self.hyp['log_ell'])
                n,D = x.shape
                if diag: D2 = np.zeros(n)
                else:    D2 = sq_dist(x,z,ell=ell)
                G = g(D2)
                K = G**(-al) * (0.5*D2/G - al*np.log(G))
            else:                                               # length scales
                K = k.evaluate(x,z=z,diag=diag,deriv=deriv-1)
        return K

class matern(Covariance):
    def __init__(self,ell=None,log_ell=None,d=1):
        """ Construct a Matérn covariance function, d=1,3,5.
        """
        if log_ell==None: log_ell = np.log(ell)
        hyp = dict(log_ell=log_ell)
        Covariance.__init__(self,hyp=hyp,name='se')
        self.d = d

    def evaluate(self, x,z=None, diag=False,deriv=None):
        """ Evaluation of Matérn covariance function.
        """
        d = self.d                                                     # degree
        if   d==1: f,df = lambda t: 1            ,lambda t: 0
        elif d==3: f,df = lambda t: 1+t          ,lambda t: 1
        elif d==5: f,df = lambda t: 1+t*(1+t/3.0),lambda t: 1+2*t/3.0
        else: raise Exception('Bad d in covariance.')
        g  = lambda t:  f(t)*np.exp(-t)
        dg = lambda t: df(t)*np.exp(-t) - g(t)
        def  h(D2): return  g(np.sqrt(d*D2))
        def dh(D2): return dg(np.sqrt(d*D2))*np.sqrt(d) / (2*np.sqrt(D2))
        k = stat(h=h,dh=dh,log_ell=self.hyp['log_ell'])
        K = k.evaluate(x,z=z,diag=diag,deriv=deriv)
        K[np.isnan(K)] = 0
        return K

class gabor(Covariance):
    def __init__(self,ell=None,log_ell=None,p=None,log_p=None):
        """ Construct a squared exponential covariance function.
        """
        if log_ell==None: log_ell = np.log(ell)
        if log_p==None: log_p = np.log(p)
        hyp = dict(log_ell=log_ell,log_p=log_p)
        Covariance.__init__(self,hyp=hyp,name='gabor')

    def evaluate(self, x,z=None, diag=False,deriv=None):
        """ Evaluation of squared exponential covariance function.
        """
        p = np.exp(self.hyp['log_p'])
        def  f(D2): return 2*np.pi/ p * np.sqrt(D2)
        def df(D2): return   np.pi/(p * np.sqrt(D2))
        def  h(D2): return np.exp(-0.5*D2)*np.cos(f(D2))
        def dh(D2): return -0.5*h(D2) - np.exp(-0.5*D2)*np.sin(f(D2)) * df(D2)
        k = stat(h=h,dh=dh,log_ell=self.hyp['log_ell'])
        if deriv==None:
            K = k.evaluate(x,z=z,diag=diag,deriv=deriv)
        else:
            if deriv==0:                           # log_p comes before log_ell
                ell = np.exp(self.hyp['log_ell'])
                n,D = x.shape
                if diag: D2 = np.zeros(n)
                else:    D2 = sq_dist(x,z,ell=ell)
                K = np.exp(-0.5*D2)*np.sin(f(D2))*f(D2)
            else:                                               # length scales
                K = k.evaluate(x,z=z,diag=diag,deriv=deriv-1)
        K[np.isnan(K)] = 0
        return K