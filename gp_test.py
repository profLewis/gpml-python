# -*- coding: utf-8 -*-
""" Test cases for the gp module also calling test cases of sub modules:
    covariance, mean, likelihood, inference.

Created: Tue Jan 07 09:20:36 2014 by Hannes Nickisch, Philips Research Hamburg.
Modified: $Id: gp_test.py 913 2013-08-15 12:54:33Z hn $
"""
__version__ = "$Id: gp_test.py 913 2013-08-15 12:54:33Z hn $"

import unittest
from test.inference  import TestInference
from test.covariance import TestCovariance
from test.mean       import TestMean
from test.likelihood import TestLikelihood

class TestGP(unittest.TestCase):

    if not hasattr(unittest.TestCase,'assertLessEqual'):
        def assertLessEqual(self,d,thr): self.assertTrue(d<thr)

    def setUp(self):
        """ Init method; called before each and every test is executed.
        """
        pass

    def tearDown(self,thr=1.5e-6):
        """ Clean up method; called after each and every test is executed.
        """
        pass

if __name__ == "__main__":     # run the test cases contained in the test suite
    test_classes = [TestInference, TestCovariance, TestMean, TestLikelihood]
    loader = unittest.TestLoader()
    if 1:
        suite = []
        for tc in test_classes: suite.append(loader.loadTestsFromTestCase(tc))
        unittest.TextTestRunner(verbosity=2).run(unittest.TestSuite(suite))
    if 0:
        for tc in test_classes:
            suite = loader.loadTestsFromTestCase(tc)
            unittest.TextTestRunner(verbosity=2).run(suite)
    if 0:
        suite = unittest.TestSuite()
        suite.addTest(TestInference("test_exact"))
        unittest.TextTestRunner(verbosity=2).run(suite)