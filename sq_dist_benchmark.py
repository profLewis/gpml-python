# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 09:10:09 2014

@author: 300227723
"""

# python -O covariance.py

import time
import numpy as np
import covariance as cov

n,m = 4096,1024
D = 12
x,z = np.random.randn(n,D),np.random.randn(m,D)
ell = np.random.rand(D)

def bench(sqd,x,z,ell):
    def d2(x,z=None,ell=None): return cov.sq_dist(x,z=z,ell=ell,sqd=sqd)
    t = time.time()
    d2xx  = d2(x,None)
    d2xz  = d2(x,z)
    d2zx  = d2(z,x)
    d2xxe = d2(x,None,ell)
    d2xze = d2(x,z,ell)
    d2zxe = d2(z,x,ell)
    print "%s.%s took %1.2fs."%(sqd.__module__,sqd.__name__,time.time()-t)
    return d2xx,d2xz,d2zx, d2xxe,d2xze,d2zxe

d2xx1,d2xz1,d2zx1, d2xxe1,d2xze1,d2zxe1 = bench(cov.sq_dist_scipy,x,z,ell=ell)
d2xx2,d2xz2,d2zx2, d2xxe2,d2xze2,d2zxe2 = bench(cov.sq_dist_dot,  x,z,ell=ell)
d2xx3,d2xz3,d2zx3, d2xxe3,d2xze3,d2zxe3 = bench(cov.sq_dist_loop, x,z,ell=ell)

dev = lambda x,y: np.max(np.abs(x-y))
err2 = dev(d2xx1,d2xx2) + dev(d2xz1,d2xz2) + dev(d2zx1,d2zx2)
erre2 = dev(d2xxe1,d2xxe2) + dev(d2xze1,d2xze2) + dev(d2zxe1,d2zxe2)
print err2,erre2
err3 = dev(d2xx1,d2xx3) + dev(d2xz1,d2xz3) + dev(d2zx1,d2zx3)
erre3 = dev(d2xxe1,d2xxe3) + dev(d2xze1,d2xze3) + dev(d2zxe1,d2zxe3)
print err3,erre3