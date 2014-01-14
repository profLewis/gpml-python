# -*- coding: utf-8 -*-
""" GPML hyper parameter treatment.

Provide iterator, getter and setter.

Created: Mon Jan 13 11:01:19 2014 by Hannes Nickisch, Philips Research Hamburg.
Modified: $Id: hyper.py 1263 2013-12-13 13:36:13Z hn $
"""
__version__ = "$Id: hyper.py 913 2013-08-15 12:54:33Z hn $"

import numpy as np

class HyperIter:
    def __init__(self):
        self.chil = []

    def __iter__(self):
        """ Iterate over hyperparameters.
        """
        for s,h in self.hyp.iteritems():
            if isinstance(h,float) or isinstance(h,int):     # scalar parameter
                yield s,None,h,self
            else:                                            # vector parameter
                for i,hi in enumerate(h): yield s,i,hi,self
        for c in self.chil:                           # recursion over children
            for y in c: yield y

    def get_hyp(self):
        """ Obtain hyperparameters as a vector.
        """
        hyp = []
        for s,i,hi,k in self: hyp.append(hi)
        return np.array(hyp)

    def set_hyp(self,hyp,j=None):
        """ Set all hyperparameters jointly or individually.
        """
        ii = 0
        for s,i,hi,k in self:
            if j==None:
                if i==None: k.hyp[s]    = hyp[ii]
                else:       k.hyp[s][i] = hyp[ii]
            elif j==ii:
                if i==None: k.hyp[s]    = hyp
                else:       k.hyp[s][i] = hyp
            ii += 1
        return self

    def get_nhyp(self):
        """ Obtain total number of hyperparameters.
        """
        return sum(1 for _ in self)

    def get_hyp_tree(self):
        """ Construct the tree of hyperparameters.
        """
        hyp_tree = []
        def dfs(k,s):                        # depth first search over the tree
            s += k.name
            if k.chil!=None:
                for i,ki in enumerate(k.chil):
                    dfs(ki,'%s%d/'%(s,i+1))
            for v,h in k.hyp.items():
                hyp_tree.append( (s,h) )
        dfs(self,'')
        return hyp_tree