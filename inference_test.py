# -*- coding: utf-8 -*-
""" Test cases for the inference module.

Created: Wed Dec 18 16:57:01 2013 by Hannes Nickisch, Philips Research Hamburg.
Modified: $Id: inference_test.py 913 2013-08-15 12:54:33Z hn $
"""
__version__ = "$Id: inference_test.py 913 2013-08-15 12:54:33Z hn $"

import numpy as np
import unittest
import inference as inf
import covariance as cov
import mean
import likelihood as lik

def erel(x,y):
    """ Relative error between two quantities.
    """
    if x==None or y==None: return None
    nx,ny = np.linalg.norm(x),np.linalg.norm(y)
    return np.linalg.norm(x-y)/max([1,nx,ny])

class TestInference(unittest.TestCase):

    if not hasattr(unittest.TestCase,'assertLessEqual'):
        def assertLessEqual(self,d,thr): self.assertTrue(d<thr)

    def setUp(self):
        """ Init method; called before each and every test is executed.
        """
        def f(x): return np.sum(x*np.sin(x),axis=1)
        np.random.seed(42)
        self.n,self.D = 12,3
        self.X = np.random.randn(self.n,self.D)
        self.y = f(self.X).ravel()

    def tearDown(self,thr=1.5e-6):
        """ Clean up method; called after each and every test is executed.
        """
        pass

    def verify_dimensions(self,im,cf,mf,lf):
        """ Check whether outputs have the right dimensions.
        """
        post,nlZ = im(cf,mf,lf,self.X,self.y)
        self.assertEqual(np.size(nlZ),1)
        self.assertEqual(post.alpha.size,self.n)
        self.assertEqual(post.sW.size,self.n)
        if post.L==None: self.assertEqual(post.C.shape,(self.n,self.n))
        if post.C==None: self.assertEqual(post.L.shape,(self.n,self.n))

    def verify_derivatives(self,inf,cov,mean,lik,thr):
        """ Compare numerical against analytical derivatives.
        """
        n,D,X,y = self.n,self.D,self.X,self.y
        post,nlZ,dnlZ = inf(cov,mean,lik,X,y,deriv=True)

        h = 1e-5
        hyp = cov.get_hyp(); dhyp = np.zeros_like(hyp)  # covariance parameters
        for i in range(len(hyp)):
            hyp[i] += h; cov.set_hyp(hyp)
            _,nlZh = inf(cov,mean,lik,X,y)
            hyp[i] -= h; cov.set_hyp(hyp)
            dhyp[i] = (nlZh-nlZ)/h
        self.assertLessEqual(erel(dhyp,dnlZ.cov),thr)
        hyp = mean.get_hyp(); dhyp = np.zeros_like(hyp)       # mean parameters
        for i in range(len(hyp)):
            hyp[i] += h; mean.set_hyp(hyp)
            _,nlZh = inf(cov,mean,lik,X,y)
            hyp[i] -= h; mean.set_hyp(hyp)
            dhyp[i] = (nlZh-nlZ)/h
        self.assertLessEqual(erel(dhyp,dnlZ.mean),thr)
        hyp = lik.get_hyp(); dhyp = np.zeros_like(hyp)  # likelihood parameters
        for i in range(len(hyp)):
            hyp[i] += h; lik.set_hyp(hyp)
            _,nlZh = inf(cov,mean,lik,X,y)
            hyp[i] -= h; lik.set_hyp(hyp)
            dhyp[i] = (nlZh-nlZ)/h
        self.assertLessEqual(erel(dhyp,dnlZ.lik),thr)

    def run_verifications(self,im,cf,mf,lf,thr=1e-4):
        """ Bundle of formal mean properties to be verified.
        """
        self.verify_dimensions(im,cf,mf,lf)
        self.verify_derivatives(im,cf,mf,lf,thr)

    def test_chol(self,thr=1e-8):
        """ Verify Cholesky decomposition.
        """
        n = 31
        A = np.random.rand(n,n)
        A = np.dot(A.T,A)
        L = inf.chol(A)
        self.assertLessEqual(np.linalg.norm(np.dot(L,L.T)-A),thr)
        self.assertLessEqual(np.linalg.norm(np.tril(L)-L),thr)

    def test_solve_chol(self,thr=1e-8):
        """ Verify Cholesky-based solution of linear equations.
        """
        n = 31
        A = np.random.rand(n,n)
        A = np.dot(A.T,A)
        B = np.random.randn(n,4)
        X = np.linalg.solve(A,B)
        Y = inf.solve_chol(inf.chol(A),B)
        self.assertLessEqual(np.linalg.norm(X-Y),thr)

    def test_exact(self,thr=1e-4):
        """ Test exact inference.
        """
        im = inf.exact
        cf,mf,lf = 1.1*cov.se(ell=0.8),mean.zero(),lik.gauss(sn=1e-5)
        self.run_verifications(im,cf,mf,lf,thr)
        lf = lik.gauss(sn=0.2)
        self.run_verifications(im,cf,mf,lf,thr)
        mf = 2.3*mean.one()
        self.run_verifications(im,cf,mf,lf,thr)
        cf = 1.3*cov.se(ell=[0.8,0.7,0.3])
        self.run_verifications(im,cf,mf,lf,thr)
        mf = 1.3*mean.linear(a=[0.8,0.7,0.3])
        self.run_verifications(im,cf,mf,lf,thr)

if __name__ == "__main__":     # run the test cases contained in the test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(TestInference)
    unittest.TextTestRunner(verbosity=2).run(suite)