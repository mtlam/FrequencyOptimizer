import unittest
import warnings
import tempfile
from os import path
from mock import patch, mock_open
import numpy as np
import parameterized as ptzd
import frequencyoptimizer as fop
from frequencyoptimizer import TelescopeNoise

"""
Unittests for frequencyoptimizer.TelescopeNoise class
"""
# enable frequency-dependent integration time for testing
fop._DISABLE_FREQDEPDT_T = False

class Test_TelescopeNoise__init__rx_nu_length_mismatch(unittest.TestCase):
    """
    Test(s) parameterized over frequency-dependent args to check whether
    ValueError is raised when frequency-dependent args
    and rx_nu lengths dont match
    """
    def setUp(self):
        self.rx_nu = np.array([1., 2., 3.])
        
    #first element of each parameter tuple must be unique
    @ptzd.parameterized.expand([("gain", np.array([1., 2.]), 1., 1., 1.),
                                ("T_rx", 1., np.array([1., 2.]), 1., 1.),
                                ("epsilon", 1., 1., np.array([1., 2.]), 1.),
                                ("t_int", 1., 1., 1., np.array([1., 2.]))],
                               name_func=lambda fxn, n, par : "_{}_".format(ptzd.parameterized.to_safe_name(par.args[0])).join(fxn.__name__.split("_arg_")))
    def test_arg_rx_nu_length_mismatch_raises_ValueError(self,
                                                      arg_name,
                                                      gain,
                                                      T_rx,
                                                      eps,
                                                      t_int):
        with self.assertRaises(ValueError):
            scope_noise = TelescopeNoise(gain,
                                         T_rx,
                                         epsilon=eps,
                                         T=t_int,
                                         rx_nu=self.rx_nu)

class Test_TelescopeNoise__init__gain_invalid_type(unittest.TestCase):
    """
    Tests parameterized over type to check whether
    TypeError raised when gain is invalid type
    """
    def setUp(self):
        self.T_rx = 1.
    
    @ptzd.parameterized.expand([("",), ([1., 2.],), (None,)],
                               name_func=lambda fxn, n, par : "_{}_".format(ptzd.parameterized.to_safe_name(str(type(par.args[0])))).join(fxn.__name__.split("_argtype_")))
    def test_gain_argtype_raises_TypeError(self, gain):
        with self.assertRaises(TypeError):
            scope_noise = TelescopeNoise(gain,
                                         self.T_rx)

class Test_TelescopeNoise__init__T_rx_invalid_type(unittest.TestCase):
    """
    Tests parameterized over type to check whether
    TypeError raised when T_rx is invalid type
    """
    def setUp(self):
        self.gain = 1.
    
    @ptzd.parameterized.expand([("",), ([1., 2.],), (None,)],
                               name_func=lambda fxn, n, par : "_{}_".format(ptzd.parameterized.to_safe_name(str(type(par.args[0])))).join(fxn.__name__.split("_argtype_")))
    def test_T_rx_argtype_raises_TypeError(self, T_rx):
        with self.assertRaises(TypeError):
            scope_noise = TelescopeNoise(self.gain,
                                         T_rx)

class Test_TelescopeNoise__init__epsilon_invalid_type(unittest.TestCase):
    """
    Tests parameterized over type to check whether
    TypeError raised when epsilon is invalid type
    """
    def setUp(self):
        self.gain = 1.
        self.T_rx = 1.
    
    @ptzd.parameterized.expand([("",), ([1., 2.],), (0,), (None,)],
                               name_func=lambda fxn, n, par : "_{}_".format(ptzd.parameterized.to_safe_name(str(type(par.args[0])))).join(fxn.__name__.split("_argtype_")))
    def test_epsilon_argtype_raises_TypeError(self, epsilon):
        with self.assertRaises(TypeError):
            scope_noise = TelescopeNoise(self.gain,
                                         self.T_rx,
                                         epsilon=epsilon)

class Test_TelescopeNoise__init__pi_V_invalid_type(unittest.TestCase):
    """
    Tests parameterized over type to check whether
    TypeError raised when pi_V is invalid type
    """
    def setUp(self):
        self.gain = 1.
        self.T_rx = 1.
    
    @ptzd.parameterized.expand([("",), ([1., 2.],), (0,), (None,)],
                               name_func=lambda fxn, n, par : "_{}_".format(ptzd.parameterized.to_safe_name(str(type(par.args[0])))).join(fxn.__name__.split("_argtype_")))
    def test_pi_V_argtype_raises_TypeError(self, pi_V):
        with self.assertRaises(TypeError):
            scope_noise = TelescopeNoise(self.gain,
                                         self.T_rx,
                                         pi_V=pi_V)
            
class Test_TelescopeNoise__init__eta_invalid_type(unittest.TestCase):
    """
    Tests parameterized over type to check whether
    TypeError raised when eta is invalid type
    """
    def setUp(self):
        self.gain = 1.
        self.T_rx = 1.
    
    @ptzd.parameterized.expand([("",), ([1., 2.],), (0,), (None,)],
                               name_func=lambda fxn, n, par : "_{}_".format(ptzd.parameterized.to_safe_name(str(type(par.args[0])))).join(fxn.__name__.split("_argtype_")))
    def test_eta_argtype_raises_TypeError(self, eta):
        with self.assertRaises(TypeError):
            scope_noise = TelescopeNoise(self.gain,
                                         self.T_rx,
                                         eta=eta)

class Test_TelescopeNoise__init__pi_L_invalid_type(unittest.TestCase):
    """
    Tests parameterized over type to check whether
    TypeError raised when pi_L is invalid type
    """
    def setUp(self):
        self.gain = 1.
        self.T_rx = 1.
    
    @ptzd.parameterized.expand([("",), ([1., 2.],), (0,), (None,)],
                               name_func=lambda fxn, n, par : "_{}_".format(ptzd.parameterized.to_safe_name(str(type(par.args[0])))).join(fxn.__name__.split("_argtype_")))
    def test_pi_L_argtype_raises_TypeError(self, pi_L):
        with self.assertRaises(TypeError):
            scope_noise = TelescopeNoise(self.gain,
                                         self.T_rx,
                                         pi_L=pi_L)

class Test_TelescopeNoise__init__T_invalid_type(unittest.TestCase):
    """
    Tests parameterized over type to check whether
    TypeError raised when T is invalid type
    """
    def setUp(self):
        self.gain = 1.
        self.T_rx = 1.
    
    @ptzd.parameterized.expand([("",), ([1., 2.],), (0,), (None,)],
                               name_func=lambda fxn, n, par : "_{}_".format(ptzd.parameterized.to_safe_name(str(type(par.args[0])))).join(fxn.__name__.split("_argtype_")))
    def test_T_argtype_raises_TypeError(self, T):
        with self.assertRaises(TypeError):
            scope_noise = TelescopeNoise(self.gain,
                                         self.T_rx,
                                         T=T)
            
class Test_TelescopeNoise__init__rx_nu_invalid_type(unittest.TestCase):
    """
    Tests parameterized over type to check whether
    TypeError raised when rx_nu is invalid type
    """
    def setUp(self):
        self.gain = 1.
        self.T_rx = 1.
    
    @ptzd.parameterized.expand([("",), ([1., 2.],), (0,), (1.,)],
                               name_func=lambda fxn, n, par : "_{}_".format(ptzd.parameterized.to_safe_name(str(type(par.args[0])))).join(fxn.__name__.split("_argtype_")))
    def test_rx_nu_argtype_raises_TypeError(self, rx_nu):
        with self.assertRaises(TypeError):
            scope_noise = TelescopeNoise(self.gain,
                                         self.T_rx,
                                         rx_nu=rx_nu)

class Test_TelescopeNoise__init__rxspecfile_invalid_type(unittest.TestCase):
    """
    Tests parameterized over type to check whether
    TypeError raised when rxspecfile is invalid type
    """
    def setUp(self):
        self.gain = 1.
        self.T_rx = 1.
    
    @ptzd.parameterized.expand([(1.,), ([1., 2.],), (0,)],
                               name_func=lambda fxn, n, par : "_{}_".format(ptzd.parameterized.to_safe_name(str(type(par.args[0])))).join(fxn.__name__.split("_argtype_")))
    def test_rxspecfile_argtype_raises_TypeError(self, rxspecfile):
        with self.assertRaises(TypeError):
            scope_noise = TelescopeNoise(self.gain,
                                         self.T_rx,
                                         rxspecfile=rxspecfile)

# class Test_TelescopeNoise__init__rxspecdir_invalid_type(unittest.TestCase):
#     """
#     Tests parameterized over type to check whether
#     TypeError raised when rxspecdir is invalid type
#     """
#     def setUp(self):
#         self.gain = 1.
#         self.T_rx = 1.
    
#     @ptzd.parameterized.expand([(1.,), ([1., 2.],), (0,)],
#                                name_func=lambda fxn, n, par : "_{}_".format(ptzd.parameterized.to_safe_name(str(type(par.args[0])))).join(fxn.__name__.split("_argtype_")))
#     def test_rxspecdir_argtype_raises_TypeError(self, rxspecdir):
#         m = mock_open()
#         m.return_value.__iter__ = lambda self: iter(self.readline, '')
#         with patch("frequencyoptimizer.open", m):
#             with self.assertRaises(TypeError):
#                 scope_noise = TelescopeNoise(self.gain,
#                                              self.T_rx,
#                                              rxspecfile="",
#                                              rxspecdir=rxspecdir)
            
class Test_TelescopeNoise__init__only_rx_nu_npndarray(unittest.TestCase):
    """
    Test whether UserWarning is raised when rx_nu is an numpy.ndarray
    but other freq-dependent params are not
    """
    def setUp(self):
        self.rx_nu = np.array([1., 2., 3.])

    @patch.object(warnings, "warn")
    def test_only_rx_nu_type_npndarray_raises_Warning(self, mock_warn):
        scope_noise = TelescopeNoise(1.,
                                     1.,
                                     epsilon=1.,
                                     T=1.,
                                     rx_nu=self.rx_nu)
        mock_warn.assert_called_once()

    @patch.object(warnings, "warn")
    def test_rx_nu_and_gain_type_npndarray_doesnt_raise_Warning(self, mock_warn):
        scope_noise = TelescopeNoise(np.array([1., 2., 3.]),
                                     1.,
                                     epsilon=1.,
                                     T=1.,
                                     rx_nu=self.rx_nu)
        self.assertFalse(mock_warn.called)

# class Test_TelescopeNoise__init__user_defined_rxspecfile_takes_precedence(unittest.TestCase):
#      """
#      Test whether user-defined rxspecfile class takes precedence over default
#      """

class Test_TelescopeNoise__init__rxspecfile_is_None(unittest.TestCase):
    """
    Test whether rx_nu, gain, T_rx, epsilon, and T attributes are the same as
    input arguments when rxspecfile is None
    """
    def setUp(self):
        self.rxspecfile = None
        
    #first element of each parameter tuple must be unique
    @ptzd.parameterized.expand([("rx_nu", np.array([1., 2.])),
                                ("gain", np.array([1., 2.])),
                                ("T_rx", np.array([1., 2.])),
                                ("epsilon", np.array([1., 2.])),
                                ("T", np.array([1., 2.]))],
                               name_func=lambda fxn, n, par : "_{}_".format(ptzd.parameterized.to_safe_name(par.args[0])).join(fxn.__name__.split("_attr_")))
    def test_attr_same_as_input_arg_when_rxspecfile_is_None(self,
                                                            attr_name,
                                                            arg_val):
        scope_noise = TelescopeNoise(arg_val,
                                     arg_val,
                                     epsilon=arg_val,
                                     T=arg_val,
                                     rx_nu=arg_val)
        self.assertIsNone(np.testing.assert_array_equal(getattr(scope_noise,
                                                                attr_name),
                                                        arg_val))

class Test_TelescopeNoise__init__nonexistent_rxspecfile(unittest.TestCase):
    """
    Test whether IOError is raised if rxspecfile is not None, but does not exist
    """
    def setUp(self):
        self.rxspecfile = ""

    @patch("frequencyoptimizer.os.path.isfile")
    def test_rxspecfile_does_not_exist_raises_IOError_errno2(self,
                                                             mock_os_path_isfile):
        mock_os_path_isfile.return_value = False
        with self.assertRaises(IOError) as err:
            scope_noise = TelescopeNoise(1.,
                                         1.,
                                         rxspecfile=self.rxspecfile)
        self.assertEqual(err.exception.errno, 2)

class Test_TelescopeNoise__init__rxspecfile_precedence(unittest.TestCase):
    """
    Test that an rxspecfile in the working directory is loaded when a file
    with the same name exists in the default rxspecs directory
    """
    def setUp(self):
        self.file_content = ("#freq	Trx	G	eps\n"
                             "1000.	10.	1.	0.01\n")
        self.basename = "test.txt"
        self.workingdir_rxspecfile = tempfile.NamedTemporaryFile(prefix=self.basename,
                                                                 dir=".")
        self.rxspecs_rxspecfile = tempfile.NamedTemporaryFile(prefix=self.basename,
                                                                 dir=path.join(fop.__dir__,
                                                                               'rxspecs'))

    def test_rxspecfile_in_working_dir_takes_precedence_over_default_dir(self):
        m = mock_open(read_data=self.file_content)
        m.return_value.__iter__ = lambda self: iter(self.readline, '')
        with patch("frequencyoptimizer.open", m):
            scope_noise = TelescopeNoise(1.,
                                         1.,
                                         rxspecfile=self.basename)
            self.assertEqual(scope_noise.rxspecfile,
                             path.abspath(path.join(".", self.basename)))

if __name__ == '__main__':
    unittest.main()
