# -*- coding: utf-8 -*-
""" Test cases for the covariance module.

Created: Wed Dec 18 16:57:01 2013 by Hannes Nickisch, Philips Research Hamburg.
Modified: $Id: covariance_test.py 913 2013-08-15 12:54:33Z hn $
"""
__version__ = "$Id: covariance_test.py 913 2013-08-15 12:54:33Z hn $"

import numpy as np
import unittest
from ..covariance import Covariance as cov

class TestCovariance(unittest.TestCase):

    if not hasattr(unittest.TestCase,'assertLessEqual'):
        def assertLessEqual(self,d,thr): self.assertTrue(d<thr)

    def setUp(self):
        """ Init method; called before each and every test is executed.
        """
        np.random.seed(42)
        self.n,self.m,self.D = 32,39,3
        self.X = np.random.randn(self.n,self.D)
        self.Z = np.random.randn(self.m,self.D)

    def tearDown(self,thr=1.5e-6):
        """ Clean up method; called after each and every test is executed.
        """
        pass

    def verify_dimensions(self,k):
        """ Check whether outputs have the right dimensions.
        """
        n,m,D,X,Z = self.n,self.m,self.D,self.X,self.Z
        K = k(X,Z)
        self.assertEqual(K.shape,(n,m))
        K = k(X)
        self.assertEqual(K.shape,(n,n))
        K = k(X,Z,diag=True)
        self.assertEqual(K.shape,(n,))
        K = k(X,diag=True)
        self.assertEqual(K.shape,(n,))

    def verify_diag(self,k,thr):
        """ Check whether the diag modifier works properly.
        """
        n,m,D,X,Z = self.n,self.m,self.D,self.X,self.Z
        K = k(X)
        dgK = np.diag(K)
        d = np.linalg.norm( dgK - k(X,diag=True) )
        self.assertLessEqual(d,thr)
        d = np.linalg.norm( dgK - k(X,Z,diag=True) )
        self.assertLessEqual(d,thr)

    def verify_derivatives(self,k,thr):
        """ Compare numerical against analytical derivatives.
        """
        n,m,D,X,Z = self.n,self.m,self.D,self.X,self.Z
        h = 1e-5
        hyp = k.get_hyp()
        Kxx,Kxz,dgK = k(X),k(X,Z),k(X,diag=True)
        for i,hi in enumerate(hyp):
            dKxx,dKxz,ddgK = k(X,deriv=i),k(X,Z,deriv=i),k(X,diag=True,deriv=i)
            k.set_hyp(hi+h,i)
            d = np.abs( (k(X)-Kxx)/h - dKxx ).max()
            self.assertLessEqual(d,thr)
            d = np.abs( (k(X,Z)-Kxz)/h - dKxz ).max()
            self.assertLessEqual(d,thr)
            d = np.abs( (k(X,diag=True)-dgK)/h - ddgK ).max()
            self.assertLessEqual(d,thr)
            k.set_hyp(hi,i)

    def run_verifications(self,k,thr=1e-6):
        """ Bundle of formal covariance properties to be verified.
        """
        self.verify_dimensions(k)
        self.verify_diag(k,thr/100)
        self.verify_derivatives(k,100*thr)

    def test_sum(self,thr=1e-6):
        """ Test the sum of several covariance functions.
        """
        n,m,D,X,Z = self.n,self.m,self.D,self.X,self.Z
        k1 = cov.se(ell=np.random.rand())
        k2 = cov.se(ell=np.random.rand(self.D))
        k = k1+k2+1.2
        d = np.linalg.norm( k1(X)+k2(X)+1.2 - k(X) )
        self.assertLessEqual(d,thr)
        self.run_verifications(k,thr)

    def test_prod(self,thr=1e-6):
        """ Test the product of several covariance functions.
        """
        n,m,D,X,Z = self.n,self.m,self.D,self.X,self.Z
        k1 = cov.se(ell=np.random.rand())
        k2 = cov.se(ell=np.random.rand(self.D))
        k = k1*k2*1.2
        d = np.linalg.norm( k1(X)*k2(X)*1.2 - k(X) )
        self.assertLessEqual(d,thr)
        self.run_verifications(k,thr)

    def test_sumprod(self,thr=1e-6):
        """ Test the sum of products of several covariance functions.
        """
        k1 = cov.se(ell=np.random.rand())
        k2 = cov.se(ell=np.random.rand(self.D))
        k3 = cov.se(ell=np.random.rand())
        k4 = cov.se(ell=np.random.rand(self.D))
        self.run_verifications(2.3*k1+k2*1.2+k3*k4,thr)

    def test_prodsum(self,thr=3e-6):
        """ Test the product of sums of several covariance functions.
        """
        k1 = cov.se(ell=np.random.rand())
        k2 = cov.se(ell=np.random.rand(self.D))
        k3 = cov.se(ell=np.random.rand())
        k4 = cov.se(ell=np.random.rand(self.D))
        self.run_verifications((2.3+k1)*(k2+1.2)*(k3+k4),thr)

    def test_noise(self,thr=1e-6):
        """ Test noise covariance.
        """
        n,m,D,X,Z = self.n,self.m,self.D,self.X,self.Z
        k = cov.noise()
        self.run_verifications(k,thr)
        d = np.linalg.norm( k(X) - np.eye(n) )
        self.assertLessEqual(d,thr)
        d = np.linalg.norm( k(X,Z) - np.eye(n,m) )
        self.assertLessEqual(d,thr)

    def test_se(self,thr=1e-6):
        """ Test squared exponential covariance.
        """
        k = cov.se(ell=np.random.rand())
        self.run_verifications(k,thr)
        k = cov.se(ell=np.random.rand(self.D))
        self.run_verifications(k,thr)

    def test_rq(self,thr=1e-6):
        """ Test rational quadratic covariance.
        """
        k = cov.rq(ell=np.random.rand(),al=1.5)
        self.run_verifications(k,thr)
        k = cov.rq(ell=np.random.rand(self.D),al=1.2)
        self.run_verifications(k,thr)

    def test_matern(self,thr=1e-6):
        """ Test Matern covariance.
        """
        for d in [1,3,5]:
            k = cov.matern(ell=np.random.rand(),d=d)
            self.run_verifications(k,thr)
            k = cov.matern(ell=np.random.rand(self.D),d=d)
            self.run_verifications(k,thr)

    def test_gabor(self,thr=1e-6):
        """ Test Gabor covariance.
        """
        k = cov.gabor(ell=np.random.rand(),p=1.5)
        self.run_verifications(k,thr)
        k = cov.gabor(ell=np.random.rand(self.D),p=1.2)
        self.run_verifications(k,thr)

if __name__ == "__main__":     # run the test cases contained in the test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(TestCovariance)
    unittest.TextTestRunner(verbosity=2).run(suite)