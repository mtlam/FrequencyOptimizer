import unittest
import warnings
from mock import patch, mock_open
import numpy as np
import parameterized as ptzd
import frequencyoptimizer as fop
from frequencyoptimizer import TelescopeNoise

"""
Unittests for frequencyoptimizer.TelescopeNoise class
"""

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
    
    @ptzd.parameterized.expand([("",), ([1., 2.],), (0,), (None,)],
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
    
    @ptzd.parameterized.expand([("",), ([1., 2.],), (0,), (None,)],
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

if __name__ == '__main__':
    unittest.main()
