import unittest
from mock import patch, mock_open
import numpy as np
import frequencyoptimizer as fop
from frequencyoptimizer import TelescopeNoise

"""
Unittests for reading and writing files
"""
class ParametrizedTestCase(unittest.TestCase):
    """ TestCase classes that want to be parametrized should
        inherit from this class.
    """
    def __init__(self, methodName='runTest', file_content=None):
        super(ParametrizedTestCase, self).__init__(methodName)
        self.file_content = file_content

    @staticmethod
    def parametrize(testcase_klass, file_content=None):
        """ Create a suite containing all tests taken from the given
            subclass, passing parameters
        """
        testloader = unittest.TestLoader()
        testnames = testloader.getTestCaseNames(testcase_klass)
        suite = unittest.TestSuite()
        for name in testnames:
            suite.addTest(testcase_klass(name,
                                         file_content=file_content))
        return suite

class Test_get_rxspecs_missing_value(ParametrizedTestCase):
    """
    Test whether an RcvrFileParseError is raised when a column is missing a value
    """
    def setUp(self):
        self.scope_noise = TelescopeNoise(1., 1.)
        self.scope_noise.rxf_path = ''

    def test_missing_values_in_rxspec_file_raises_RcvrFileParseError(self):
        print("file_content =\n{}".format(self.file_content))
        m = mock_open(read_data=self.file_content)
        m.return_value.__iter__ = lambda self: iter(self.readline, '')
        with patch("frequencyoptimizer.open", m):
            with self.assertRaisesRegexp(fop.RcvrFileParseError, ".*must have 4 or 5.*"):
                self.scope_noise.get_rxspecs(0.)
                print(self.scope_noise.get_rxspecs(0.))

class Test_get_rxspecs_blank_line(unittest.TestCase):
    """
    Test that get_rxspecs skips blank line in file without halting execution
    """
    def setUp(self):
        self.scope_noise = TelescopeNoise(1., 1.)
        self.scope_noise.rxf_path = ''
        self.file_content = ("# Header metadata\n" 
                             "#freq	Trx	G	eps     t_int\n"
                             ".422	42.	10.	0.01    1.\n"
                             "\n"
                             ".424      22.	7.5	0.01    1.")
        self.correct_answer = (np.array([.422, .424]),
                               np.array([42., 22.]),
                               np.array([10., 7.5]),
                               np.array([0.01, 0.01]),
                               np.array([1., 1.]))

    def test_rxspecfile_blank_line_doesnt_raise_RcvrFileParseError(self):
        print("file_content =\n{}".format(self.file_content))        
        m = mock_open(read_data=self.file_content)
        m.return_value.__iter__ = lambda self: iter(self.readline, '')
        with patch("frequencyoptimizer.open", m):
            np.testing.assert_allclose(self.scope_noise.get_rxspecs(0.),
                                       self.correct_answer)

                
class Test_get_rxspecs_t_int_is_array(unittest.TestCase):
    """
    Test whether a TypeError is raised when receiver specs file is specified
    and tint_in is an array
    """
    def setUp(self):
        self.scope_noise = TelescopeNoise(1., 1.)
        self.scope_noise.rxf_path = ''
        self.file_content = ("# Header metadata\n" 
                             "#freq	Trx	G	eps \n"
                             ".422	42.	10.	0.01\n"
                             ".424      22.	7.5	0.01")
        self.tint_in = np.array([1800., 3600.])

    def test_rxspecfile_is_not_None_and_tint_in_is_array_raises_TypeError(self):
        print("tint_in = {}".format(self.tint_in))
        m = mock_open(read_data=self.file_content)
        m.return_value.__iter__ = lambda self: iter(self.readline, '')
        with patch("frequencyoptimizer.open", m):
            with self.assertRaises(TypeError):
                self.scope_noise.get_rxspecs(self.tint_in)
                
class Test_get_rxspecs_invalid_header(ParametrizedTestCase):
    """
    Test whether an RcvrFileParseError is raised when file does not contain
    or contains an invalid header
    """
    def setUp(self):
        self.scope_noise = TelescopeNoise(1., 1.)
        self.scope_noise.rxf_path = ''

    def test_invalid_header_in_rxspec_file_raises_RcvrFileParseError(self):
        print("file_content =\n{}".format(self.file_content))
        m = mock_open(read_data=self.file_content)
        m.return_value.__iter__ = lambda self: iter(self.readline, '')
        with patch("frequencyoptimizer.open", m):
            with self.assertRaisesRegexp(fop.RcvrFileParseError,
                                         ".*missing.*header.*"):
                self.scope_noise.get_rxspecs(0.)    
        
if __name__ == '__main__':
    suite = unittest.TestSuite()
    get_rxspecs_missing_val_test_params = [("# Header metadata\n" 
                                            "#freq	Trx	G	eps \n"
                                            ".422	42.	10.	0.01\n"
                                            "	        22.	7.5	0.01"),
                                           ("# Header metadata\n" 
                                            "#freq	Trx	G	eps \n"
                                            ".422	42.	10.	0.01\n"
                                            ".424		7.5	0.01"),
                                           ("# Header metadata\n" 
                                            "#freq	Trx	G	eps \n"
                                            ".422	42.	10.	0.01\n"
                                            ".424	22.	7.5	")]
    suite.addTest(unittest.makeSuite(Test_get_rxspecs_blank_line))
    for fstr in get_rxspecs_missing_val_test_params:
        suite.addTest(ParametrizedTestCase.parametrize(Test_get_rxspecs_missing_value,
                                                       file_content=fstr))
    get_rxspecs_invalid_header_test_params = [("# Header metadata\n" 
                                               ".422	42.	10.	0.01\n"
                                               ".424	22.	7.5	0.01"),
                                              (".422	42.	10.	0.01\n"
                                               ".424	22.	7.5	0.01"),
                                              ("# Header metadata\n"
                                               "#Trx	freq	G	eps \n"
                                               ".422	42.	10.	0.01\n"
                                               ".424	22.	7.5	0.01"),
                                              ("# Header metadata\n"
                                               "#freq	Tsys	G	eps \n"
                                               ".422	42.	10.	0.01\n"
                                               ".424	22.	7.5	0.01")]
    for fstr in get_rxspecs_invalid_header_test_params:
        suite.addTest(ParametrizedTestCase.parametrize(Test_get_rxspecs_invalid_header,
                                                       file_content=fstr))
    suite.addTest(unittest.makeSuite(Test_get_rxspecs_t_int_is_array))
    unittest.TextTestRunner(verbosity=2).run(suite)

