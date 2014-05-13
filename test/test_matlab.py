# -*- coding: utf-8 -*-
"""
Created on Fri May 09 16:22:39 2014

@author: 300227723
"""

import scipy.io as sio

mat = sio.loadmat('regression.mat')
nlZ = float(mat['nlZ'])
class dnlZ: mean,cov,lik = mat['dnlZ'][0][0]
class post: L,alpha,sW   = mat['post'][0][0]
class hyp:  mean,cov,lik = mat['hyp'] [0][0]
x,y = mat['x'],mat['y']
# lf,mf,cf need to be done manually

mat = sio.loadmat('classification.mat')
nlZ = float(mat['nlZ'])
class dnlZ: mean,cov,lik = mat['dnlZ'][0][0]
class post: L,alpha,sW   = mat['post'][0][0]
class hyp:  mean,cov,lik = mat['hyp'] [0][0]
x,y = mat['x'],mat['y']
# lf,mf,cf need to be done manually