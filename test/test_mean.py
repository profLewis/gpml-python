# -*- coding: utf-8 -*-
""" Test cases for the mean module.

Created: Wed Dec 18 16:57:01 2013 by Hannes Nickisch, Philips Research Hamburg.
Modified: $Id: mean_test.py 913 2013-08-15 12:54:33Z hn $
"""
__version__ = "$Id: mean_test.py 913 2013-08-15 12:54:33Z hn $"

if __name__ == "__main__" and __package__ is None:  # make parent dir available
    import sys,os
    sys.path.insert(0, os.path.abspath('..'))

import numpy as np
import unittest
import mean

class TestMean(unittest.TestCase):

    if not hasattr(unittest.TestCase,'assertLessEqual'):
        def assertLessEqual(self,d,thr): self.assertTrue(d<thr)

    def setUp(self):
        """ Init method; called before each and every test is executed.
        """
        np.random.seed(42)
        self.n,self.D = 12,3
        self.X = np.random.randn(self.n,self.D)

    def tearDown(self,thr=1.5e-6):
        """ Clean up method; called after each and every test is executed.
        """
        pass

    def verify_dimensions(self,m):
        """ Check whether outputs have the right dimensions.
        """
        n,D,X = self.n,self.D,self.X
        M = m(X)
        self.assertEqual(M.shape,(n,))

    def verify_derivatives(self,m,thr):
        """ Compare numerical against analytical derivatives.
        """
        n,D,X = self.n,self.D,self.X
        h = 1e-5
        hyp = m.get_hyp()
        M = m(X)
        for i,hi in enumerate(hyp):
            dM = m(X,deriv=i)
            m.set_hyp(hi+h,i)
            d = np.abs( (m(X)-M)/h - dM ).max()
            self.assertLessEqual(d,thr)
            m.set_hyp(hi,i)

    def run_verifications(self,m,thr=1e-6):
        """ Bundle of formal mean properties to be verified.
        """
        self.verify_dimensions(m)
        self.verify_derivatives(m,100*thr)

    def test_sum(self,thr=1e-6):
        """ Test the sum of several meanfunctions.
        """
        n,D,X = self.n,self.D,self.X
        m1 = mean.one()
        m2 = mean.linear(a=np.random.rand(self.D))
        m = m1+m2+1.2
        d = np.linalg.norm( m1(X)+m2(X)+1.2 - m(X) )
        self.assertLessEqual(d,thr)
        self.run_verifications(m,thr)

    def test_prod(self,thr=1e-6):
        """ Test the product of several mean functions.
        """
        n,D,X = self.n,self.D,self.X
        m1 = mean.one()
        m2 = mean.linear(a=np.random.rand(self.D))
        m = m1*m2*1.7
        d = np.linalg.norm( m1(X)*m2(X)*1.7 - m(X) )
        self.assertLessEqual(d,thr)
        self.run_verifications(m,thr)

    def test_pow(self,thr=1e-6):
        """ Test power of a mean function.
        """
        n,D,X = self.n,self.D,self.X
        m1 = mean.one()
        m2 = mean.linear(a=np.random.rand(self.D))
        m = 1.2*m1 * m2**2
        d = np.linalg.norm( 1.2*m1(X)*m2(X)**2 - m(X) )
        self.assertLessEqual(d,thr)
        self.run_verifications(m,thr)

    def test_sumprod(self,thr=1e-6):
        """ Test the sum of products of several mean functions.
        """
        m1 = mean.one()
        m2 = mean.linear(a=np.random.rand(self.D))
        m3 = mean.one()
        m4 = mean.linear(a=np.random.rand(self.D))
        self.run_verifications(2.3*m1+m2*1.2+m3*m4,thr)

    def test_prodsum(self,thr=1e-6):
        """ Test the product of sums of several mean functions.
        """
        m1 = mean.one()
        m2 = mean.linear(a=np.random.rand(self.D))
        m3 = mean.one()
        m4 = mean.linear(a=np.random.rand(self.D))
        self.run_verifications((2.3+m1)*(m2+1.2)*(m3+m4),thr)

    def test_zero(self,thr=1e-6):
        """ Test zero mean.
        """
        n,D,X = self.n,self.D,self.X
        m = mean.zero()
        self.run_verifications(m,thr)
        d = np.linalg.norm( m(X) )
        self.assertLessEqual(d,thr)

    def test_one(self,thr=1e-6):
        """ Test one mean.
        """
        n,D,X = self.n,self.D,self.X
        m = mean.one()
        self.run_verifications(m,thr)
        d = np.linalg.norm( m(X)-1.0 )
        self.assertLessEqual(d,thr)

    def test_linear(self,thr=1e-6):
        """ Test one mean.
        """
        n,D,X = self.n,self.D,self.X
        m = mean.linear(a=np.random.randn(D))
        self.run_verifications(m,thr)

    def test_poly(self,thr=1e-6):
        """ Test polynomial mean.
        """
        n,D,X = self.n,self.D,self.X
        A = np.random.randn(4,D)
        m = mean.poly(A=A)
        self.run_verifications(m,thr)

if __name__ == "__main__":     # run the test cases contained in the test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(TestMean)
    unittest.TextTestRunner(verbosity=2).run(suite)