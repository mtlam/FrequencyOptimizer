import os
import unittest
import timeit
from itertools import product
import numpy as np
import frequencyoptimizer as fop
from DISS import scale_dnu_d

class ParametrizedTestCase(unittest.TestCase):
    """ TestCase classes that want to be parametrized should
        inherit from this class.
    """
    def __init__(self, methodName='runTest', screen=None, fresnel=None):
        super(ParametrizedTestCase, self).__init__(methodName)
        self.screen = screen
        self.fresnel = fresnel

    @staticmethod
    def parametrize(testcase_klass, screen=None, fresnel=None):
        """ Create a suite containing all tests taken from the given
            subclass, passing screen and fresnel.
        """
        testloader = unittest.TestLoader()
        testnames = testloader.getTestCaseNames(testcase_klass)
        suite = unittest.TestSuite()
        for name in testnames:
            suite.addTest(testcase_klass(name, screen=screen,
                                         fresnel=fresnel))
        return suite

class Test_build_DMnu_covmat(ParametrizedTestCase):

    def runTest(self):
        pass

    def setUp(self):
        ''' Set up frequency optimizer instance '''
        bw = 0.6 # GHz
        n_channels = 100
        ctrfreq = 1.4
#        self.nus = np.array([1.1, 1.16, 1.4])
#        ctrfreq = np.median(self.nus)
        self.nus = np.linspace(ctrfreq - bw / 2, ctrfreq + bw / 2, n_channels + 1)[:-1]
        self.psr_noise = fop.PulsarNoise('',
                                         alpha=1.28,
                                         dtd=0.048,
                                         dnud=10.,
                                         taud=6865638.25,
                                         C1=1.16,
                                         I_0=0.0309,
                                         DM=770.89,
                                         D=12.502,
                                         tauvar=0.5 * 6865638.25,
                                         Weffs=819.73,
                                         W50s=171.95,
                                         Uscale=44.56,
                                         sigma_Js=0.366,
                                         glon=-0.7056,
                                         glat=37.0666)
        self.scope_noise = fop.TelescopeNoise(2.,
                                              T_const=22.73,
                                              T=1800,
                                              epsilon=0.01)
        self.gal_noise = fop.GalacticNoise()
        self.fop_inst = fop.FrequencyOptimizer(self.psr_noise,
                                               self.gal_noise,
                                               self.scope_noise,
                                               nchan=len(self.nus),
                                               numax=ctrfreq,
                                               numin=ctrfreq,
                                               vverbose=False)

    def test_new_covmatrix_matches_old(self):
        print("screen = {}, fresnel = {}".format(self.screen, self.fresnel))

        # time computation of covmat using old method
        start1 = timeit.default_timer()
        for i in range(200):
            old_covmat = self.build_DMnu_cov_matrix_old(screen=self.screen,
                                                        fresnel=self.fresnel)
        stop1 = timeit.default_timer()
        t1 = stop1 - start1

        # time computation of covmat under development
        start2 = timeit.default_timer()
        for i in range(200):
            new_covmat = self.fop_inst.build_DMnu_cov_matrix(self.nus,
                                                             screen=self.screen,
                                                             fresnel=self.fresnel)
        stop2 = timeit.default_timer()
        t2 = stop2 - start2
        
        np.testing.assert_allclose(new_covmat, old_covmat,
                                   rtol=1e-10)
        self.assertIsInstance(new_covmat, np.matrix)
        print("Refactor is faster by {}x".format(t1/t2))

    def build_DMnu_cov_matrix_old(self, g=0.46, q=1.15,
                                  screen=False, fresnel=False, nuref=1.0):
        '''
        [deprecated] Constructs the frequency-dependent DM error covariance
        matrix using old for-loop method to compare with numpy matrix method
        '''

        dnud = scale_dnu_d(self.psr_noise.dnud, nuref, self.nus)

        # Construct the matrix, this could be sped up by a factor of two
        retval = np.matrix(np.zeros((len(self.nus), len(self.nus))))
        for i in range(len(self.nus)):
            for j in range(len(self.nus)):

                if self.nus[i] == self.nus[j]:
                    continue # already set to zero

                #nu2 should be less than nu1
                if self.nus[i] > self.nus[j]: 
                    nu1 = self.nus[i]
                    nu2 = self.nus[j]
                    dnuiss = dnud[i]
                else:
                    nu1 = self.nus[j]
                    nu2 = self.nus[i]
                    dnuiss = dnud[j]
                 
                sigma = self.evalDMnuError_old(dnuiss, nu1, nu2, g=g, q=q,
                                               screen=screen, fresnel=fresnel)
                retval[i, j] = sigma**2
        return retval

        
    def evalDMnuError_old(self, dnuiss, nu1, nu2, g=0.46, q=1.15,
                          screen=False, fresnel=False):
        '''[deprecated] old method, calculate DMnu between two frequencies'''
        if screen:
            g = 1
        if fresnel:
            phiF = dnuiss
        else:
            phiF = 9.6 * ((nu1 / dnuiss)/100)**(5.0/12) #equation 15
        r = nu1/nu2
        sigma = 0.184 * g * q * fop.E_beta(r) * (phiF**2 / (nu1 * 1000)) 
        return sigma 

@unittest.skip("""Full keyword has been removed in current version. May return in later version.""")
class Test_DM_misestimation(unittest.TestCase):
    ''' Make sure speedup doesn't break other uses of evalDMnuError'''
    
    def runTest(self):
        pass

    def setUp(self):
        ''' Set up frequency optimizer instance '''
        bw = 0.6 # GHz
        n_channels = 100
        ctrfreq = 1.4
#        self.nus = np.array([1.1, 1.16, 1.4])
#        ctrfreq = np.median(self.nus)
        self.nus = np.linspace(ctrfreq - bw / 2, ctrfreq + bw / 2, n_channels + 1)[:-1]
        self.psr_noise = fop.PulsarNoise('',
                                         alpha=1.28,
                                         dtd=0.048,
                                         dnud=10.,
                                         taud=6865638.25,
                                         C1=1.16,
                                         I_0=0.0309,
                                         DM=770.89,
                                         D=12.502,
                                         tauvar=0.5 * 6865638.25,
                                         Weffs=819.73,
                                         W50s=171.95,
                                         Uscale=44.56,
                                         sigma_Js=0.366,
                                         glon=-0.7056,
                                         glat=37.0666)
        self.scope_noise = fop.TelescopeNoise(2.,
                                              T_const=22.73,
                                              T=1800,
                                              epsilon=0.01)
        self.gal_noise = fop.GalacticNoise()
        self.fop_inst = fop.FrequencyOptimizer(self.psr_noise,
                                               self.gal_noise,
                                               self.scope_noise,
                                               nchan=len(self.nus),
                                               numax=ctrfreq,
                                               numin=ctrfreq,
                                               full=False,
                                               vverbose=False)
        sncov = self.fop_inst.build_template_fitting_cov_matrix(self.nus)
        jittercov = jittercov = self.fop_inst.build_jitter_cov_matrix()
        disscov = self.fop_inst.build_scintillation_cov_matrix(self.nus)
        self.cov = sncov + jittercov + disscov

    def test_full_is_False(self):
        dm_miserr = self.fop_inst.DM_misestimation(self.nus, self.cov, covmat=True)
        self.assertIsInstance(dm_miserr, float)

if __name__ == '__main__':
    suite = unittest.TestSuite()
    for screen, fresnel in product(*[(True, False)]*2):
        suite.addTest(ParametrizedTestCase.parametrize(Test_build_DMnu_covmat,
                                                       screen=screen,
                                                       fresnel=fresnel))
    suite.addTest(unittest.makeSuite(Test_DM_misestimation))
    unittest.TextTestRunner(verbosity=2).run(suite)
