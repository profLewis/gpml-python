# -*- coding: utf-8 -*-
""" GPML Mean functions.

Mean values to be used in a Gaussian process.

Created: Wed Dec 18 16:57:01 2013 by Hannes Nickisch, Philips Research Hamburg.
Modified: $Id: mean.py 1263 2013-12-13 13:36:13Z hn $
"""
__version__ = "$Id: mean.py 913 2013-08-15 12:54:33Z hn $"

# TODO: mask

import numpy as np
from hyper import HyperIter

class Mean(HyperIter):
    """ Mean function base class.
    """
    def __init__(self,hyp={},name='plain',chil=[]):
        self.hyp = hyp
        self.name,self.chil = name,chil

    def __str__(self):
        return "mean/"+self.name

    def __repr__(self):
        return self.__str__()

    def __mul__(self, other):
        """ Product of a mean function and a scalar or another
            mean function.
        """
        if isinstance(other,float) or isinstance(other,int):
            hyp = dict(c=other)
            other = Mean(hyp=hyp,name='const')
        if self.name=='prod':    # make sure all terms appear on the same level
            self.chil.append(other)
            return self
        else:
            return Mean(name='prod',chil=[other,self])
    def __rmul__(self, other):                      # other is not of type Mean
        return self.__mul__(other)

    def __pow__(self, other):
        if not isinstance(other,float) and not isinstance(other,int):
            raise Exception('Only numerical powers allowed.')
        m = Mean(name='pow',chil=[self])
        m.d = other
        return m

    def __add__(self, other):
        """ Sum of a mean function and a scalar or another
            covariance function.
        """
        if isinstance(other,float) or isinstance(other,int):
            hyp = dict(c=other)
            other = Mean(hyp=hyp,name='const')
        if self.name=='sum':     # make sure all terms appear on the same level
            self.chil.append(other)
            return self
        else:
            return Mean(name='sum',chil=[other,self])
    def __radd__(self, other):                      # other is not of type Mean
        return self.__add__(other)

    def evaluate(self,x, deriv=None):
        """ Default evaluation function to be overridden by derived classes.
        """
        n = x.shape[0]
        if self.name in ['prod','sum']:
            if deriv==None:                                          # evaluate
                if   self.name=='prod':
                    M = np.ones(n)
                    for m in self.chil: M *= m(x)
                elif self.name=='sum' :
                    M = np.zeros(n)
                    for m in self.chil: M += m(x)
            else:                                        # evaluate derivatives
                deriv = int(deriv)                           # derivative index
                if deriv<0 or deriv>self.get_nhyp()-1:
                    raise Exception('Bad derivative index.')
                nhyp = np.cumsum([c.get_nhyp() for c in self.chil])
                j = np.nonzero(deriv<nhyp)[0].min()             # index of term
                if j==0: i = deriv                          # index inside term
                else:    i = deriv-nhyp[j-1]
                if   self.name=='prod':
                    M = np.ones(n)
                    for l,m in enumerate(self.chil):
                        if l==j: M *= m(x,deriv=i)
                        else:    M *= m(x)
                elif self.name=='sum' :
                    M = self.chil[j](x,deriv=i)
        elif self.name=='const':
            if deriv==None: M = self.hyp['c']*np.ones(n)
            else:           M = np.ones(n)
        elif self.name=='pow':
            m,d = self.chil[0],self.d
            if deriv==None: M = m(x)**d
            else:           M = (d * m(x)**(d-1)) * m(x,deriv=deriv)
        else:                                                    # unknown name
            M = np.zeros(n)
        return M

    def __call__(self,x,deriv=None):
        return self.evaluate(x,deriv=deriv)

def sq_dist(x,z):
    """ Compute a matrix of all pairwise squared distances
        between two sets of vectors, stored in the row of the two matrices:
        x (of size n by D) and z (of size m by D).
    """
    x = x.T
    if z==None: z = x
    else:       z = z.T
    (D,n),m = x.shape,z.shape[1]
    C = np.zeros([n,m])
    for d in range(D): C += (x[d].reshape(n,1)-z[d].reshape(1,m))**2
    return C

class zero(Mean):
    def __init__(self):
        """ Construct a zero mean function.
        """
        Mean.__init__(self,hyp={},name='zero')

    def evaluate(self,x,deriv=None):
        """ Evaluation of zero mean function.
        """
        return np.zeros(x.shape[0])

class one(Mean):
    def __init__(self):
        """ Construct a one mean function.
        """
        Mean.__init__(self,hyp={},name='one')

    def evaluate(self,x,deriv=None):
        """ Evaluation of one mean function.
        """
        if deriv==None: return np.ones( x.shape[0])
        else:           return np.zeros(x.shape[0])

class linear(Mean):
    def __init__(self,a=None):
        """ Construct a linear mean function.
        """
        Mean.__init__(self,dict(a=np.array(a)),name='linear')

    def evaluate(self,x,deriv=None):
        """ Evaluation of linear mean function.
        """
        n,D = x.shape
        a = self.hyp['a'].flatten()
        if deriv==None: return np.dot(x,a)
        else:           return x[:,deriv]

class poly(Mean):
    def __init__(self,A=None):
        """ Construct a polynomial mean function.
        """
        Mean.__init__(self,dict(A=np.array(A).flatten()),name='poly')

    def evaluate(self,x,deriv=None):
        """ Evaluation of polynomial mean function.
        """
        n,D = x.shape
        A = self.hyp['A'].reshape(-1,D)
        if deriv==None:
            m = np.zeros(n)
            for i,ai in enumerate(A): m += np.dot(x**i,ai)
            return m
        else:
            return x[:,deriv%D]**int(deriv/D)