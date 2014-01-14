# -*- coding: utf-8 -*-
""" Test cases for the likelihood module.

Created: Mon Jan 13 15:39:35 2014 by Hannes Nickisch, Philips Research Hamburg.
Modified: $Id: likelihood_test.py 913 2013-08-15 12:54:33Z hn $
"""
__version__ = "$Id: likelihood_test.py 913 2013-08-15 12:54:33Z hn $"

import numpy as np
import unittest
import likelihood as lik

def erelmax(x,y):
    """ Maximum relative error between two quantities.
    """
    if x==None or y==None: return None
    m1xy = np.maximum(1,np.maximum(np.abs(x),np.abs(y)))
    return np.max( np.abs(x-y)/m1xy )

class TestLikelihood(unittest.TestCase):

    if not hasattr(unittest.TestCase,'assertLessEqual'):
        def assertLessEqual(self,d,thr): self.assertTrue(d<thr)

    def setUp(self):
        """ Init method; called before each and every test is executed.
        """
        np.random.seed(42)
        n = 5
        self.fmu = 3*np.random.randn(n)
        self.fs2 = (0.1+3*np.random.rand(n))**2
        self.ycont = 3*np.random.randn(n)
        self.ybin = np.sign(np.random.randn(n))

    def tearDown(self,thr=1.5e-6):
        """ Clean up method; called after each and every test is executed.
        """
        pass

    def verify_la_derivatives(self,lik,y,thr=1e-3):
        """ Laplace approximation derivatives verification.
        """
        try:
            h = 1e-5
            lp,dlp,d2lp,d3lp = lik.la(y,self.fmu)
            lp_h,dlp_h,d2lp_h,_ = lik.la(y,self.fmu+h)
            dlpn,d2lpn,d3lpn = (lp_h-lp)/h,(dlp_h-dlp)/h,(d2lp_h-d2lp)/h
            self.assertLessEqual(erelmax(dlp,dlpn),thr)
            self.assertLessEqual(erelmax(d2lp,d2lpn),thr)
            self.assertLessEqual(erelmax(d3lp,d3lpn),thr)
            hyp = lik.get_hyp()
            for i in range(len(hyp)):
                lp_dh,dlp_dh,d2lp_dh = lik.la(y,self.fmu,i=i)
                hyp[i] += h; lik.set_hyp(hyp)
                lp_h,dlp_h,d2lp_h,_ = lik.la(y,self.fmu)
                hyp[i] -= h; lik.set_hyp(hyp)
                lp_dhn,dlp_dhn = (lp_h-lp)/h,(dlp_h-dlp)/h
                d2lp_dhn = (d2lp_h-d2lp)/h
                self.assertLessEqual(erelmax(lp_dh,lp_dhn),thr)
                self.assertLessEqual(erelmax(dlp_dh,dlp_dhn),thr)
                self.assertLessEqual(erelmax(d2lp_dh,d2lp_dhn),thr)
        except NotImplementedError,err: print err
        except: raise

    def verify_ep_derivatives(self,lik,y,thr=1e-3):
        """ Expectation propagation derivatives verification.
        """
        try:
            h = 1e-5
            lZ,dlZ,d2lZ = lik.ep(y,self.fmu,self.fs2)
            lZ_h,dlZ_h,d2lZ_h = lik.ep(y,self.fmu+h,self.fs2)
            dlZn,d2lZn = (lZ_h-lZ)/h,(dlZ_h-dlZ)/h
            self.assertLessEqual(erelmax(dlZ,dlZn),thr)
            self.assertLessEqual(erelmax(d2lZ,d2lZn),thr)
            hyp = lik.get_hyp()
            for i in range(len(hyp)):
                lZ_dh = lik.ep(y,self.fmu,self.fs2,i=i)
                hyp[i] += h; lik.set_hyp(hyp)
                lZ_h,_,_ = lik.ep(y,self.fmu,self.fs2)
                hyp[i] -= h; lik.set_hyp(hyp)
                lZ_dhn = (lZ_h-lZ)/h
                self.assertLessEqual(erelmax(lZ_dh,lZ_dhn),thr)
        except NotImplementedError,err: print err
        except: raise

    def run_verifications(self,likfun,y,thr=1e-3):
        """ Run different verification tests.
        """
        self.verify_la_derivatives(likfun,y,thr=thr)
        self.verify_ep_derivatives(likfun,y,thr=thr)
    
    def test_gauss(self):
        """ Verify Gaussian likelihood.
        """
        self.run_verifications(lik.gauss(sn=0.1),self.ycont)
    
    def test_erf(self):
        """ Verify error function likelihood.
        """
        self.run_verifications(lik.erf(),self.ybin)

if __name__ == "__main__":     # run the test cases contained in the test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(TestLikelihood)
    unittest.TextTestRunner(verbosity=2).run(suite)