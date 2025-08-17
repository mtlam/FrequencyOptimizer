import numpy as np
import scipy.linalg as linalg
import scipy.interpolate as interpolate
import scipy.optimize as optimize
from matplotlib.pyplot import *
from matplotlib import cm,rc
from matplotlib.ticker import FuncFormatter, MultipleLocator
import matplotlib.patches as patches
import DISS
import warnings
import parallel
import os

# temporarily disable frequency-dependent integration time until simultaneous
# multi-band is fully supported
_DISABLE_FREQDEPDT_T = True

__dir__ = os.path.dirname(os.path.abspath(__file__))


np.seterr(invalid="warn")


rc('text',usetex=True)
rc('font',**{'family':'serif','serif':['Times New Roman'],'size':14})#,'weight':'bold'})
rc('xtick',**{'labelsize':16})
rc('ytick',**{'labelsize':16})
rc('axes',**{'labelsize':18,'titlesize':18})


def nolog(x,pos):
    return r"$\hfill %0.1f$" % (10**x)
noformatter = FuncFormatter(nolog)
def nolog2(x,pos):
    return r"$\hfill %0.2f$" % (10**x)
noformatter2 = FuncFormatter(nolog2)

def log(x,pos):
    y = x#np.log10(x)
    #if y == 2:
    #    return "$\hfill 100$" #added
    if y == 1:
        return r"$\hfill 10$"
    elif y == 0:
        return r"$\hfill 1$"
    elif y == -1:
        return r"$\hfill 0.1$"
    elif y == -2:
        return r"$\hfill 0.01$"
    return r"$\hfill 10^{%i}$" % x#np.log10(x) 

formatter = FuncFormatter(log)



def log100(x,pos):
    y = x#np.log10(x)
    if y == 2:
        return r"$\hfill 100$" #added
    elif y == 1:
        return r"$\hfill 10$"
    elif y == 0:
        return r"$\hfill 1$"
    elif y == -1:
        return r"$\hfill 0.1$"
    elif y == -2:
        return r"$\hfill 0.01$"
    return r"$\hfill 10^{%i}$" % x#np.log10(x) 

formatter100 = FuncFormatter(log100)

# Copied from utilities.py
def uimshow(x,ax=None,origin='lower',interpolation='nearest',aspect='auto',**kwargs):
    if ax is not None:
        im=ax.imshow(x,origin=origin,interpolation=interpolation,aspect=aspect,**kwargs)
    else:
        im=imshow(x,origin=origin,interpolation=interpolation,aspect=aspect,**kwargs) # plt.
    return im









#K = 4.149 #ms GHz^2 pc^-1 cm^3
K = 4.149e3 #us GHz^2 pc^-1 cm^3  
# Note on units used: TOA errors in microseconds, observing frequencies in GHz, DM in pc cm^-3

LEVELS = np.array([np.log10(0.125),np.log10(0.25),np.log10(0.5),np.log10(1.0)])
LEVELS = np.array([np.log10(0.25),np.log10(0.5),np.log10(1.0),np.log10(2.0),np.log10(5.0)])
LEVELS = np.array([np.log10(0.25),np.log10(0.5),np.log10(1.0),np.log10(2.0),np.log10(5.0)])
LEVELS = np.array([np.log10(0.5),np.log10(1.0),np.log10(2.0),np.log10(5.0),np.log10(10.0)])
LEVELS = np.array([np.log10(0.1),np.log10(0.2),np.log10(0.5),np.log10(1.0),np.log10(2.0)])
#LEVELS = np.array([np.log10(0.5),np.log10(1.0),np.log10(2.0),np.log10(5.0),np.log10(10.0),np.log10(20.0)])

COLORS = ['k','0.25','0.5','0.75']
COLORS = ['k','0.25','0.5','0.75','1.0']
#COLORS = ['k','0.2','0.4','0.6','0.8','1.0']

LWS = [2.5,2,1.5,1,0.5]
LWS = [2.5,2.25,2,1.75,1.5]
#LWS = [2.5,2.25,2.0,1.75,1.5,1.25]

RXFILE_HEADER_FMT = "#Freq  Trx  G  Eps  t_int(optional)"

def epoch_averaged_error(C,var=False):
    # Stripped down version from rednoisemodel.py from the excess noise project
    N = len(C)
    UT = np.matrix(np.ones(N))
    U = UT.T
    try:
        CI = C.I
    except np.linalg.LinAlgError:
        print("Warning: singular matrix, using pseudoinverse")
        CI = np.linalg.pinv(C)
    C_E = np.dot(np.dot(UT,CI),U).I
    if var:
        return C_E[0,0]
    return np.sqrt(C_E[0,0])







def evalNonSimError(dtiss,nu1,nu2,tau):
    # dtiss at 1 GHz, tau in days
    # Returns error in microseconds
    # Equation 14
    return 6.5e-3 * abs(1.0/(nu1**2 - nu2**2)) * (tau / (dtiss/1000))**(5.0/6)

# DMnu-related variables
def F_beta(r,beta=11.0/3):
    return np.sqrt(2**((4-beta)/2.0) * (1 + r**((2*beta)/(beta-2)))**((beta-2)/2.0) - r**beta - 1)
def E_beta(r,beta=11.0/3):
    r2 = r**2
    return np.abs(r2 / (r2-1)) * F_beta(r,beta)


def evalDMnuError(dnud,nus,g=0.46,q=1.15,screen=False,fresnel=False):
    ''' Returns matrix of DMnu errors'''
    # nu2 should be less than nu1
    # nu in GHz, dnuiss in GHz
    # return value in microseconds
    # Based on equation 25
    # if fresnel==True, the first argument is phiF
    if screen:
        g = 1
    if fresnel:
        phiF = dnud
    else:
        phiF = 9.6 * ((nus / dnud)/100)**(5.0/12) #equation 15
    r = np.outer(1 / nus, nus)
    np.fill_diagonal(r, 0)
    sig_asym = 0.184 * g * q * E_beta(r) * (phiF**2 / (nus * 1000))
    #nu2 should be < nu1 so lower triangle should = upper
    sigma = np.triu(sig_asym) + np.triu(sig_asym).transpose()
    return sigma

def readtskyfile():
    """Read in tsky.ascii into a list from which temps can be retrieved."""

    tskypath = os.path.join(__dir__, 'lookuptables/tsky.ascii')
    tskylist = []
    with open(tskypath) as f:
        for line in f:
            str_idx = 0
            while str_idx < len(line):
                # each temperature occupies space of 5 chars
                temp_string = line[str_idx:str_idx+5]
                try:
                    tskylist.append(float(temp_string))
                except:
                    pass
                str_idx += 5

    return tskylist

def tskypy(tskylist, gl, gb, freq):
    """ Calculate tsky from Haslam table, scale to frequency in MHz."""
    # ensure l is in range 0 -> 360
    b = gb
    if gl < 0.:
        l = 360 + gl
    else:
        l = gl
        
    # convert from l and b to list indices
    j = b + 90.5
    if j > 179:
        j = 179
            
    nl = l - 0.5
    if l < 0.5:
        nl = 359
    i = float(nl) / 4.

    tsky_haslam = tskylist[180*int(i) + int(j)]
    # scale temperature before returning
    return tsky_haslam * (freq/408.0)**(-2.6)

def get_rxspecs_options():
    return os.listdir(os.path.join(__dir__, 'rxspecs'))

class PulsarNoise:
    '''
    Container class for all pulsar-related variables

    alpha: Pulsar flux spectral index
    dtd: Scintillation timescale (s)
    dnud: Scintillation bandwidth (GHz)
    taud: Scattering timescale (us)
    C1: Coefficient relating dnud to taud (1.16 = uniform Kolmogorov medium) 
    I_0: Pulsar flux density at 1 GHz
    DM: Dispersion measure (pc cm^-3)
    D: Distance (kpc)
    Uscale: Dimensionless factor that describes how intensity is distributed across pulse phase, see Sec. 2.2.1 of (Lam et al. 2018)
    tauvar: Variation in scattering timescale (us)
    Weffs: Effective width, can be an array (us)
    W50s: Pulse full-width at half-maximum, can be an array (us)
    sigma_Js: Jitter for observation time T, can be an array (us) [note: T needs to be related to the TelescopeNoise class]
    P: float (optional)
       spin period in ms, if supplied sets upper limit on sigmas
    glon: Galactic longitude (deg)
    glat: galactic latitude (deg)
    ampratios_file: string (optional, default="ampratios.npz")
                    Name of pulse broadening function data numpy zip archive
    or path to user-defined file. A user-defined file path takes precedence over default files. NpzFile must contain keys ['errratios', 'Weffratios', 'ratios', 'ampratios'].
    '''
    def __init__(self,name,alpha=1.6,dtd=None,dnud=None,taud=None,C1=1.16,I_0=18.0,DM=0.0,D=1.0,Uscale=1.0,tauvar=None,Weffs=0.0,W50s=0.0,sigma_Js=0.0,P=None,glon=None,glat=None, ampratios_file="ampratios.npz"):
        self.name = name
        self.glon = glon
        self.glat = glat
        self.dnud = dnud
        self.taud = taud

        if dtd is None:
            #Assume dtd is large?
            self.dtd = 10000.0
        else:
            self.dtd = dtd

        if taud is None and dnud is None:
            # Assume taud is 0 and dnud is very large
            self.taud = 0.0
            self.dnud = 10000.0
        elif taud is None:
            self.dnud = dnud
            self.taud = 1e-3 * C1/(2*np.pi*dnud) #taud0 in ns -> us
        elif dnud is None:
            self.taud = taud
            self.dnud = 1e-3 * C1/(2*np.pi*taud) #taud0 given in us, dnud0 in GHz

        '''
        if taud is not None:
            self.taud = taud #taud now in us
            self.dnud = C1 / (2*np.pi*taud) #if dnud in GHz, taud in ns
        elif dnud is not None:
            self.dnud = dnud
            self.taud = C1 / (2*np.pi*dnud)
        '''

        self.C1 = C1
        self.I_0 = I_0
        self.DM = DM
        self.D = D

        self.alpha = alpha

        if tauvar is None:
            tauvar = self.taud / 2.0
        self.tauvar = tauvar

        self.Weffs = Weffs
        self.W50s = W50s
        self.sigma_Js = sigma_Js
        self.Uscale = Uscale

        if P is not None:
            self.P = P * 1000 # now in microseconds
        else:
            self.P = None

        self.load_ampratios_data(ampratios_file)
            
    def load_ampratios_data(self, ampratios_file):
        # load pulse broadening function data
        if os.path.isfile(ampratios_file):
            self.ampratios_file = ampratios_file
            ampratios_npz = np.load(self.ampratios_file)
        elif os.path.isfile(os.path.join(__dir__, ampratios_file)): # default
            self.ampratios_file = os.path.join(__dir__,
                                               ampratios_file)
            ampratios_npz = np.load(self.ampratios_file)
        else:
            raise IOError(2, "'ampratios_file' does not exist. ",
                          ampratios_file)
        required_columns = ['errratios', 'Weffratios', 'ratios', 'ampratios']
        try:
            self.ampratios_data = {'ampratios': ampratios_npz['ampratios'],
                                   'ratios': ampratios_npz['ratios'],
                                   'Weffratios': ampratios_npz['Weffratios'],
                                   'errratios': ampratios_npz['errratios']}
        except KeyError:
            raise KeyError("NpzFile 'ampratios_file' must contain "
                           "keys {}.".format(required_columns))

class GalacticNoise:
    '''
    Container class for all Galaxy-related variables.

    beta: Galactic-background spectral index
    T_e (K) [deprecated]: Electron temperature
    fillingfactor [deprecated]: Filling factor of electrons
    '''
    def __init__(self,beta=2.75,T_e=100.0,fillingfactor=0.2):
        self.beta = beta
        self.T_e = T_e
        self.fillingfactor = fillingfactor
        self.tskylist = readtskyfile()


class RcvrFileParseError(Exception):
    pass

class TelescopeNoise:
    '''
    Container class for all Telescope-related variables.

    gain : int, float or numpy.ndarray
           Telescope gain (K/Jy) 
           If array must be same length as rx_nu 
    T_rx : int, float or numpy.ndarray
           Receiver temperature (K) (i.e. T_sys - T_gal - T_CMB)
           If array must be same length as rx_nu 
    epsilon : float or numpy.ndarray (optional)
              Fractional gain error
              If array must be same length as rx_nu
    pi_V : float (optional) 
           Degree of circular polarization
    eta : float (optional)
          Voltage cross-coupling coefficient
    pi_L : float (optional)
           Degree of linear polarization
    T : float (optional)
        Integration time (s)
    Npol : int or float (optional)
           Number of polarization states
    rx_nu : None or numpy.ndarray (optional)
            Receiver frequencies over which to interpolate (GHz)
    rxspecfile : string (optional)
                 Name of receiver specifications file or path to user-defined file. A user-defined file takes precedence over default files. I.e. A file in the working directory will override a default file with the same name. Call frequencyoptimizer.get_rxspecs_options() to see default files.
                 If defined, a file overrides gain, T_rx, and epsilon arguments. Files must contain a header with the format

                #Freq  Trx  G  Eps

                immediately followed by 4 tab-separated columns of frequency, T_rx, gain, and epsilon.
    '''
    def __init__(self, gain, T_rx, epsilon=0.08,
                 pi_V=0.1, eta=0.0, pi_L=0.0,
                 T=1800.0, Npol=2, rx_nu=None,
                 rxspecfile=None, rxspecdir=None):

        if not isinstance(gain, (float, int, np.ndarray)):
            raise TypeError("Invalid 'gain' type {}. Valid types are float, int, "
                            "or numpy.ndarray.".format(type(gain)))
        if isinstance(gain, int):
            gain = float(gain)
        if isinstance(gain, np.ndarray):
            try:
                if len(gain) != len(rx_nu):
                    raise ValueError("'gain' and 'rx_nu' must be "
                                     "the same length.")
            except TypeError:
                raise TypeError("if 'gain' is type numpy.ndarray, "
                                "rx_nus must also be numpy.ndarray of same length")
        if not isinstance(T_rx, (float, int, np.ndarray)):
            raise TypeError("Invalid 'T_rx' type {}. Valid types are float, int, "
                            "or numpy.ndarray.".format(type(T_rx)))
        if isinstance(T_rx, int):
            T_rx = float(T_rx)
        if isinstance(T_rx, np.ndarray):
            try:
                if len(T_rx) != len(rx_nu):
                    raise ValueError("'T_rx' and 'rx_nu' must be "
                                     "the same length.")
            except TypeError:
                raise TypeError("if 'T_rx' is type numpy.ndarray, "
                                "rx_nus must also be numpy.ndarray of same length")
        if not isinstance(epsilon, (float, np.ndarray)):
            raise TypeError("Invalid 'epsilon' type {}. Valid types are float "
                            "or numpy.ndarray.".format(type(epsilon)))
        if isinstance(epsilon, np.ndarray):
            try:
                if len(epsilon) != len(rx_nu):
                    raise ValueError("'epsilon' and 'rx_nu' must be "
                                     "the same length.")
            except TypeError:
                raise TypeError("if 'epsilon' is type numpy.ndarray, "
                                "rx_nus must also be numpy.ndarray of same length")
        if not isinstance(pi_V, float):
            raise TypeError("Invalid 'pi_V' type {}. Valid types are "
                            "float.".format(type(pi_V)))
        if not isinstance(eta, float):
            raise TypeError("Invalid 'eta' type {}. Valid types are "
                            "float.".format(type(eta)))
        if not isinstance(pi_L, float):
            raise TypeError("Invalid 'pi_L' type {}. Valid types are "
                            "float.".format(type(pi_L)))
        if isinstance(T, np.ndarray):
            if _DISABLE_FREQDEPDT_T:
                raise NotImplementedError("Frequency-dependent T not "
                                          "fully supported")
            try:
                if len(T) != len(rx_nu):
                    raise ValueError("'T' and 'rx_nu' must be "
                                     "the same length.")
            except TypeError:
                raise TypeError("if 'T' is type numpy.ndarray, "
                                "rx_nus must also be numpy.ndarray of same length")
        elif not isinstance(T, float):
            raise TypeError("Invalid 'T' type {}. Valid types are float "
                            "or numpy.ndarray.".format(type(T)))
        if not isinstance(Npol, (int, float)):
            raise TypeError("Invalid 'Npol' type {}. Valid types are int or "
                            "float.".format(type(Npol)))
        if not isinstance(rx_nu, (type(None), np.ndarray)):
            raise TypeError("Invalid 'rx_nu' type {}. Valid types are None or "
                            "numpy.ndarray.".format(type(rx_nu)))
        if isinstance(rx_nu, np.ndarray) and not any([isinstance(k, np.ndarray)\
                                                      for k in [gain,
                                                                T_rx,
                                                                epsilon,
                                                                T]]):
            warnings.warn("rx_nu is type numpy.ndarray but other "
                          "frequency-dependent parameters are not. "
                          "Ignoring rx_nu and not interpolating.")
        if not isinstance(rxspecfile, (type(None), str)):
            raise TypeError("Invalid 'rxspecfile' type {}. Valid types are None "
                            "or str.".format(type(rxspecfile)))
        
        if rxspecfile is None:
            self.rxspecfile = rxspecfile
            self.rx_nu = rx_nu
            self.T_rx = T_rx
            self.gain = gain
            self.epsilon = epsilon
            self.T = T
        elif os.path.isfile(rxspecfile):
            self.rxspecfile = os.path.abspath(rxspecfile)
            self.rx_nu, self.T_rx, self.gain, self.epsilon, self.T = self.get_rxspecs(T)
        elif os.path.isfile(os.path.join(__dir__, 'rxspecs', rxspecfile)):
            self.rxspecfile = os.path.abspath(os.path.join(__dir__,
                                                           'rxspecs',
                                                           rxspecfile))
            self.rx_nu, self.T_rx, self.gain, self.epsilon, self.T = self.get_rxspecs(T)
        else:
            raise IOError(2, "'rxspecfile' does not exist. ", rxspecfile)

        self.pi_V = pi_V
        self.eta = eta
        self.pi_L = pi_L
        self.Npol = Npol
                
    def get_gain(self,nu):
        if isinstance(self.gain, np.ndarray):
            return np.interp(nu,self.rx_nu,self.gain)
        else:
            return self.gain
    def get_epsilon(self,nu):
        if isinstance(self.epsilon, np.ndarray):
            return np.interp(nu,self.rx_nu,self.epsilon)
        else:
            return self.epsilon
    def get_T_rx(self,nu):
        if isinstance(self.T_rx, np.ndarray):
            return np.interp(nu,self.rx_nu,self.T_rx)
        else:
            return self.T_rx
    def get_T(self,nu):
        if isinstance(self.T, np.ndarray):
            return np.interp(nu,self.rx_nu,self.T)
        else:
            return self.T
    def get_rxspecs(self, tint_in):
        with open(self.rxspecfile, 'r') as rxf:
            rx_nus = []
            trxs = []
            gains = []
            eps = []
            t_ints = []
            header_requires = ['freq', 'trx', 'g', 'eps']
            is_header = lambda l : l.startswith('#') and l.strip("#").lower().split()[:4] == header_requires
            # read file
            for line in rxf:
                if is_header(line): # find header
                    header = line
                    for line in rxf: # read data (lines after header)
                        if not line.strip(): # ignore blanks
                            continue
                        lsp = line.split()
                        try:
                            rx_nus.append(float(lsp[0]))
                            trxs.append(float(lsp[1]))
                            gains.append(float(lsp[2]))
                            eps.append(float(lsp[3]))
                        except IndexError:
                            raise RcvrFileParseError("Receiver specifications file "
                                                     "must have 4 or 5 "
                                                     "columns of even length. "
                                                     "Format is\n" + RXFILE_HEADER_FMT)
                        try:
                            t_ints.append(float(lsp[4]))
                        except IndexError:
                            pass
                else:
                    header = None
            if header is None:
                raise RcvrFileParseError("Receiver specifications file "
                                         "has missing or invalid header. "
                                         "Format is\n" +
                                         RXFILE_HEADER_FMT)
            # if no t_int column
            if len(t_ints) == 0:
                if not isinstance(tint_in, (int, float)):
                    raise TypeError("If receiver specifications file "
                                    "does not contain a "
                                    "'t_int' column, 'T' must be of type "
                                    "int or float, "
                                    "not {}".format(type(tint_in)))
                else:
                    t_ints = np.full(len(rx_nus), tint_in)
        # if not all([len(l) == len(rx_nus) for l in [trxs, gains, eps, t_ints]]):
        #     # might be redundant
        #     raise RcvrFileParseError("Columns in receiver specifications file are "
        #                              "of uneven length.")
        return (np.array(rx_nus), np.array(trxs), np.array(gains), np.array(eps),
                np.array(t_ints))



class FrequencyOptimizer:
    '''
    Primary class for frequency optimization

    psrnoise: Pulsar Noise object
    galnoise: Galaxy Noise object
    telnoise: Telescope Noise object
    numin: Lowest frequency to run (GHz)
    numax: Highest frequency to run (GHz)
    nsteps: Number of steps in the grid to run when log=True
    dnu: Grid spacing when log=False
    nchan: number of underlying frequency channels
    log: Run in log space
    frac_bw: Run in fractional bandwidth
    full_bandiwdth: enforce full bandwidth in calculations
    r: maximum bandwidth ratio, for max(nus)/min(nus) > r sigmas set to NaN
    masks: mask frequencies [not fully implemented]
    levels: contour levels
    colors: contour colors
    lws: contour linewidths
    ncpu: number of cores for multiprocess threading
    '''
    
    def __init__(self,psrnoise,galnoise,telnoise,numin=0.01,numax=10.0,r=None,dnu=0.05,nchan=100,log=False,nsteps=8,frac_bw=False,verbose=True,vverbose=False,full_bandwidth=False,masks=None,levels=LEVELS,colors=COLORS,lws=LWS,ncpu=1):



        self.psrnoise = psrnoise
        self.galnoise = galnoise
        self.telnoise = telnoise
        self.log = log
        self.frac_bw = frac_bw
        self.r = r

        
        self.numin = numin
        self.numax = numax
        self.masks = masks
        if type(masks) == tuple: #implies it is not None
            self.masks = [masks]



        if self.frac_bw == False:
            if self.log == False:
                self.dnu = dnu
                self.Cs = np.arange(numin,numax,dnu)
                self.Bs = np.arange(numin,numax/2,dnu)
            else:
                MIN = np.log10(numin)
                MAX = np.log10(numax)
                self.Cs = np.logspace(MIN,MAX,int((MAX-MIN)*nsteps+1))
                if full_bandwidth:
                    MAX = np.log10(2*numax)
                    self.Bs = np.logspace(MIN,MAX,int((MAX-MIN)*nsteps+1))
                else:
                    self.Bs = np.logspace(MIN,MAX,int((MAX-MIN)*nsteps+1))
        else:
            if self.log == False:
                pass
            else:
                MIN = np.log10(numin)
                MAX = np.log10(numax)
                self.Cs = np.logspace(MIN,MAX,int((MAX-MIN)*nsteps+1))
                self.Bs = np.logspace(MIN,MAX,int((MAX-MIN)*nsteps+1))
                self.Fs = np.logspace(np.log10(self.Bs[-1]/self.Cs[0]),np.log10(1.0),len(self.Cs))[::-1]
                self.Fs = np.logspace(np.log10(self.Bs[0]/self.Cs[-1]),np.log10(2.0),len(self.Cs))
                # do not log space?
                self.Fs = np.linspace(self.Bs[0]/self.Cs[-1],2.0,len(self.Cs))



        self.nchan = nchan

        self.scattering_mod_f = None
        self.verbose = verbose
        if vverbose:
            self.verbose = True
        self.vverbose = vverbose
        self.levels = levels
        self.colors = colors
        self.lws = lws
        self.ncpu = ncpu

    def template_fitting_error(self,S,Weff=100.0,Nphi=2048): #Weff in microseconds
        return Weff / (S * np.sqrt(Nphi))



    def get_bandwidths(self,nus):
        if self.log == False:
            # assume equal bins?
            B = np.diff(nus)[0]
            #B = np.concatenate((np.diff(nus),self.dnu))
        else:
            logdiff = np.diff(np.log10(nus))[0]
            edges = 10**(np.concatenate(([np.log10(nus[0])-logdiff/2.0],np.log10(nus)+logdiff/2.0)))
            B = np.diff(edges)
        return B


    def build_template_fitting_cov_matrix(self,nus,nuref=1.0):
        '''
        Constructs the template-fitting error (i.e., from finite signal-to-noise ratio) covariance matrix
        '''
        
        Weffs = self.psrnoise.Weffs
        if type(Weffs) != np.ndarray:
            Weffs = np.zeros_like(nus)+Weffs
        B = self.get_bandwidths(nus)
       
        if self.psrnoise.glon is None or self.psrnoise.glat is None:
            Tgal = 20*np.power(nus/0.408,-1*self.galnoise.beta)
        else:
            Tgal = np.array([tskypy(self.galnoise.tskylist,
                                    self.psrnoise.glat,
                                    self.psrnoise.glon,
                                    nu*1e3) for nu in nus])
        Tsys = self.telnoise.get_T_rx(nus) + Tgal + 2.73

        
        tau = 0.0
        if self.psrnoise.DM != 0.0 and self.psrnoise.D != 0.0 and self.galnoise.T_e != 0.0 and self.galnoise.fillingfactor != 0:
            tau = 1.417e-6 * (self.galnoise.fillingfactor/0.2)**-1 * self.psrnoise.DM**2 * self.psrnoise.D**-1 * np.power(self.galnoise.T_e/100,-1.35)

        numer =  (self.psrnoise.I_0 * 1e-3) * np.power(nus/nuref,-1*self.psrnoise.alpha)*np.sqrt(self.telnoise.Npol*B*1e9*self.telnoise.get_T(nus)) 
        #* np.exp(-1*tau*np.power(nus/nuref,-2.1)) #

        denom = Tsys / self.telnoise.get_gain(nus)
        S = self.psrnoise.Uscale*numer/denom # numer/denom is the mean S/N over all phase. Need to adjust by the factor Uscale.

        
        #print numer,denom

        #print nus,B
        #print self.psrnoise.I_0,self.telnoise.gain,B,self.telnoise.get_T(nus)#np.power(nus/nuref,-1*self.psrnoise.alpha)
        
        sigmas = self.template_fitting_error(S,Weffs,1)

        if self.psrnoise.taud > 0.0:
            tauds = DISS.scale_tau_d(self.psrnoise.taud,nuref,nus)
            retval = self.scattering_modifications(tauds,
                                                   Weffs,
                                                   self.psrnoise.ampratios_data)
            #retval = 1
            sigmas *= retval #??

        # Any enormous values should not cause an overflow
        inds = np.where(sigmas>1e100)[0]
        sigmas[inds] = 1e100


        # implement masks here
        if self.masks is not None:
            for i,mask in enumerate(self.masks):
                maskmin,maskmax = mask
                inds = np.where(np.logical_and(nus>=maskmin,nus<=maskmax))[0]
                sigmas[inds] = 0.0 #???
        
        return np.matrix(np.diag(sigmas**2))

    def build_jitter_cov_matrix(self, nus):
        '''
        Constructs the jitter error covariance matrix
        '''
        sigma_Js = self.psrnoise.sigma_Js
        if type(sigma_Js) != np.ndarray:
            sigma_Js = np.zeros(len(nus), dtype=nus.dtype) + sigma_Js
        retval = np.matrix(np.outer(sigma_Js, sigma_Js))

        return retval


    def scattering_modifications(self,tauds,Weffs,data):
        '''
        Takes the calculations of the convolved Gaussian-exponential simulations and returns the multiplicative factor applies to the template-fitting errors
        '''
        if type(Weffs) != np.ndarray:
            Weffs = np.zeros_like(nus)+Weffs

        if self.scattering_mod_f is None:
            ratios = data['ratios']
            ampratios = data['ampratios']
            Weffratios = data['Weffratios']
            errratios = data['errratios']
            
            logratios = np.log10(ratios)
            logerrratios = np.log10(errratios)

            self.scattering_mod_f = interpolate.interp1d(logratios,logerrratios)

        dataratios = np.array(tauds)/np.array(Weffs) #sigma_Ws?

        retval = np.zeros_like(dataratios) + 1.0
        inds = np.where(dataratios > 0.01)[0] #must be greater than this value
        retval[inds] = 10**self.scattering_mod_f(np.log10(dataratios[inds]))
        return retval
        


    def build_scintillation_cov_matrix(self,nus,nuref=1.0,C1=1.16,etat=0.2,etanu=0.2):
        '''
        Constructs the scintillation (finite-scintle effect) error covariance matrix
        '''

        numin = nus[0]
        numax = nus[-1]

        B = self.get_bandwidths(nus)
        dtd = DISS.scale_dt_d(self.psrnoise.dtd,nuref,nus)
        dnud = DISS.scale_dnu_d(self.psrnoise.dnud,nuref,nus)
        taud = DISS.scale_tau_d(self.psrnoise.taud,nuref,nus)

        niss = (1 + etanu* B/dnud) * (1 + etat* self.telnoise.get_T(nus)/dtd) 

        # check if niss >> 1?
        sigmas = taud/np.sqrt(niss)

        retval = np.matrix(np.diag(sigmas**2))

        inds = np.where(niss < 2)[0]
        for i in inds:
            for j in inds:
                retval[i,j] = sigmas[i] * sigmas[j] #close enough?
        return retval
        #return np.matrix(np.diag(sigmas**2)) #these will be independent IF niss is large
        
        



    # Using notation from signal processing notes, lecture 17
    def DM_misestimation(self,nus,errs,covmat=False):#,fullDMnu=True):
        '''
        Return sum of DM mis-estimation errors
        '''
        N = len(nus)
        X = np.matrix(np.ones((N,2))) #design matrix
        for i,nu in enumerate(nus):
            X[i,1] = K/nu**2

        # Template-Fitting Errors
        if covmat is False:
            V = np.matrix(np.diag(errs**2)) #weights matrix
        else:
            V = errs
        XT = X.T
        VI = V.I
        P = np.dot(np.dot(XT,VI),X).I 




        # for now, ignore covariances and simply return the t_inf error    
        template_fitting_var = P[0,0] 

        ## Frequency-Dependent DM
        #DM_nu_var = evalDMnuError(self.psrnoise.dnud,np.max(nus),np.min(nus))**2 / 25.0
        DM_nu_cov = self.build_DMnu_cov_matrix(nus)
        DM_nu_var = epoch_averaged_error(DM_nu_cov,var=True)
        #print nus
        # FOO
        #print DM_nu_cov
        #print DM_nu_var
        if DM_nu_var < 0.0:# or np.isnan(DM_nu_var): #no longer needed
            DM_nu_var = 0 


        

        # PBF errors (scattering), included already in cov matrix?
        # Scattering error, assume this is proportional to nu^-4.4? or 4?
        chromatic_components = self.psrnoise.tauvar * np.power(nus,-4.4)
        scattering_var = np.dot(np.dot(np.dot(P,XT),VI),chromatic_components)[0,0]**2




        retval = np.sqrt(template_fitting_var + DM_nu_var + scattering_var)
        
        if self.vverbose:
            print("DM misestimation noise: %0.3f us"%retval)
            
            print("   DM estimation error: %0.3f us"%np.sqrt(template_fitting_var))
            print("   DM(nu) error: %0.3f us"%np.sqrt(DM_nu_var))
            print("   Chromatic term error: %0.3f us"%np.sqrt(scattering_var))


        return retval



    def build_DMnu_cov_matrix(self,nus,g=0.46,q=1.15,screen=False,fresnel=False,nuref=1.0):
        '''
        Constructs the frequency-dependent DM error covariance matrix
        '''
        dnud = DISS.scale_dnu_d(self.psrnoise.dnud,nuref,nus)
        sigma = evalDMnuError(dnud,nus,g=g,q=q,screen=screen,fresnel=fresnel)
        return np.asmatrix(sigma**2)

    def build_polarization_cov_matrix(self,nus):
        '''
        Constructs the polarization error covariance matrix
        '''
        W50s = self.psrnoise.W50s
        if type(W50s) != np.ndarray:
            W50s = np.zeros(self.nchan)+W50s
        #if type(self.telnoise.get_epsilon(nus)) != np.ndarray:
        #    epsilon = np.zeros(self.nchan)+self.telnoise.get_epsilon(nus)
        if type(self.telnoise.pi_V) != np.ndarray:
            pi_V = np.zeros(self.nchan)+self.telnoise.pi_V
        if type(self.telnoise.eta) != np.ndarray:
            eta = np.zeros(self.nchan)+self.telnoise.eta
        if type(self.telnoise.pi_L) != np.ndarray:
            pi_L = np.zeros(self.nchan)+self.telnoise.pi_L


        epsilon = self.telnoise.get_epsilon(nus)
        sigmas = epsilon*pi_V*(W50s/100.0) #W50s in microseconds #do more?
        sigmasprime = 2 * np.sqrt(eta) * pi_L #Actually use this
        return np.matrix(np.diag(sigmas**2))


    def calc_single(self,nus,retall=False):
        '''
        Calculate sigma_TOA given a selection of frequencies
        '''
        sncov = self.build_template_fitting_cov_matrix(nus)

        jittercov = self.build_jitter_cov_matrix(nus) #needs to have same length as nus!
        disscov = self.build_scintillation_cov_matrix(nus) 

        cov = sncov + jittercov + disscov

        sigma2 = epoch_averaged_error(cov,var=True)

        sigmasn2 = epoch_averaged_error(sncov, var=True)

        if self.vverbose:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                print("White noise: %0.3f us"%np.sqrt(sigma2))
                print("   Template-fitting error: %0.3f us"%np.sqrt(epoch_averaged_error(sncov,var=True)))
                if np.all(jittercov == jittercov[0,0]):
                    print("   Jitter error: %0.3f us"%np.sqrt(jittercov[0,0]))
                else:
                    print("   Jitter error: %0.3f us"%np.sqrt(epoch_averaged_error(jittercov,var=True)))
                if np.all(disscov == disscov[0,0]):
                    print("   Scintillation error: %0.3f us"%np.sqrt(disscov[0,0]))
                else:
                    print("   Scintillation error: %0.3f us"%np.sqrt(round(epoch_averaged_error(disscov,var=True),6)))


        
        
        sigmatel2 = epoch_averaged_error(self.build_polarization_cov_matrix(nus))


        sigmadm2 = self.DM_misestimation(nus,cov,covmat=True)**2


        sigma = np.sqrt(sigmadm2 + sigmatel2) #need to include PBF errors?

        if self.vverbose:
            print("Telescope noise: %0.3f us"%np.sqrt(sigmatel2))


        if self.vverbose:
            print("Total noise: %0.3f us"%sigma)
            print("")

        if self.psrnoise.P is not None and sigma > self.psrnoise.P:
            return (self.psrnoise.P,) * 5

            
        return sigma, np.sqrt(sigma2), np.sqrt(sigmadm2), np.sqrt(sigmatel2),\
            np.sqrt(sigmasn2)


    def calc(self):
        '''
        Run a full calculation over a grid of frequencies
        '''
        print("Computing for pulsar: %s"%self.psrnoise.name)
        self.sigmas = np.zeros((len(self.Cs),len(self.Bs)))
        if self.frac_bw == False:
            def loop_func(ic):
                C = self.Cs[ic]
                sigmas = np.zeros(len(self.Bs))
                if self.verbose:
                    print("Computing center freq %0.3f GHz (%i/%i)"%(C,ic,len(self.Cs)))
                for ib,B in enumerate(self.Bs):
                    #print C,B
                    #if B > 1.9*C:
                    #if B > 2*C*(self.r - 1)/(self.r + 1):
                    if (self.r is not None and (C+0.5*B)/(C-0.5*B) > self.r)\
                        or B > 1.9*C or C - B/2.0 < self.numin:
                        sigmas[ib] = np.nan
                    else:
                        nulow = C - B/2.0
                        nuhigh = C + B/2.0

                        if self.log == False:
                            nus = np.linspace(nulow,nuhigh,self.nchan+1)[:-1] #more uniform sampling?
                        else:
                            nus = np.logspace(np.log10(nulow),np.log10(nuhigh),self.nchan+1)[:-1] #more uniform sampling?
                        try:
                            sigmas[ib] = self.calc_single(nus)[0]
                        except TypeError as e:
                            print(self.calc_single(nus))
                            raise e
                        #print(self.sigmas[ic,ib])
                return sigmas

        else:
            def loop_func(ic):
                C = self.Cs[ic]
                sigmas = np.zeros(len(self.Fs))
                if verbose:
                    print(ic,len(self.Cs),C)
                for indf,F in enumerate(self.Fs):
                    B = C*F
                    if B > 1.9*C or B <= 0:
                        sigmas[indf] = np.nan
                    else:
                        nulow = C - B/2.0
                        nuhigh = C + B/2.0


                        if self.log == False:
                            nus = np.linspace(nulow,nuhigh,self.nchan+1)[:-1] #more uniform sampling?
                        else:
                            nus = np.logspace(np.log10(nulow),np.log10(nuhigh),self.nchan+1)[:-1] #more uniform sampling?   

                        sigmas[indf] = self.calc_single(nus)[0]
                return sigmas

        if self.ncpu == 1:
            for ic,C in enumerate(self.Cs):
                self.sigmas[ic,:] = loop_func(ic)
        else: #should set export OPENBLAS_NUM_THREADS=1
            if self.verbose:
                print("Attempting multiprocessing, nprocs=%s"%str(self.ncpu))
            self.sigmas[:,:] = parallel.parmap(loop_func,range(len(self.Cs)),nprocs=self.ncpu)


    def plot(self,filename="triplot.png",doshow=True,figsize=(8,6),save=True,minimum=None,points=None,colorbararrow=None,cmap=cm.inferno_r,**kwargs):
        '''
        Create the triangle plots as in the optimal frequencies paper.

        filename: filename
        doshow: show the plot
        figsize = figure size
        save: Save the figure
        minimum: Symbol to place over the minimum
        points: Place other points on the plot
        '''
        fig = figure(figsize=figsize)
        ax = fig.add_subplot(111)
        if self.frac_bw == False:
            data = np.transpose(np.log10(self.sigmas))
            if self.log == False:
                im = uimshow(data,extent=[self.Cs[0],self.Cs[-1],self.Bs[0],self.Bs[-1]],cmap=cmap,ax=ax,**kwargs)

                ax.set_xlabel(r"$\mathrm{Center~Frequency~\nu_0~(GHz)}$")
                ax.set_ylabel(r"$\mathrm{Bandwidth}~B~\mathrm{(GHz)}$")
            else:

                im = uimshow(data,extent=np.log10(np.array([self.Cs[0],self.Cs[-1],self.Bs[0],self.Bs[-1]])),cmap=cmap,ax=ax,**kwargs)
                cax = ax.contour(data,extent=np.log10(np.array([self.Cs[0],self.Cs[-1],self.Bs[0],self.Bs[-1]])),colors=self.colors,levels=self.levels,linewidths=self.lws,origin='lower')

                #https://stackoverflow.com/questions/18390068/hatch-a-nan-region-in-a-contourplot-in-matplotlib
                # get data you will need to create a "background patch" to your plot
                xmin, xmax = ax.get_xlim()
                ymin, ymax = ax.get_ylim()
                xy = (xmin,ymin)
                width = xmax - xmin
                height = ymax - ymin
                # create the patch and place it in the back of countourf (zorder!)
                p = patches.Rectangle(xy, width, height, hatch='X', color='0.5', fill=None, zorder=-10)
                ax.add_patch(p)


                ax.set_xlabel(r"$\mathrm{Center~Frequency~\nu_0~(GHz)}$")
                ax.set_ylabel(r"$\mathrm{Bandwidth}~B~\mathrm{(GHz)}$")
                ax.xaxis.set_major_locator(MultipleLocator(0.5))
                ax.yaxis.set_major_locator(MultipleLocator(0.5))
                ax.xaxis.set_major_formatter(noformatter)
                ax.yaxis.set_major_formatter(noformatter)

                ax.text(0.05,0.9,"PSR~%s"%self.psrnoise.name.replace("-","$-$"),fontsize=18,transform=ax.transAxes,bbox=dict(boxstyle="square",fc="white"))

            if minimum is not None:
                checkdata = np.log10(self.sigmas)
                flatdata = checkdata.flatten()
                #inds = np.where(np.logical_not(np.isnan(flatdata)))[0]
                inds = np.where((~np.isnan(flatdata))&~(np.isinf(flatdata)))[0]
                MIN = np.min(flatdata[inds])
                INDC,INDB = np.where(checkdata==MIN)
                INDC,INDB = INDC[0],INDB[0]
                MINB = self.Bs[INDB]
                MINC = self.Cs[INDC]
                cax = ax.contour(data,extent=np.log10(np.array([self.Cs[0],self.Cs[-1],self.Bs[0],self.Bs[-1]])),colors=['b','b'],levels=[np.log10(1.1*(10**MIN)),np.log10(1.5*(10**MIN))],linewidths=[1,1],linestyles=['--','--'],origin='lower')
                print("Minimum",MINC,MINB,MIN)
                with open("minima.txt",'a') as FILE:
                    FILE.write("%s minima %f %f %f\n"%(self.psrnoise.name,MINC,MINB,MIN))
                if self.log:
                    ax.plot(np.log10(MINC),np.log10(MINB),minimum,zorder=50,ms=10)
                else:
                    ax.plot(MINC,MINB,minimum,zorder=50,ms=10)

            if points is not None:
                if type(points) == tuple:
                    points = [points]
                for point in points:
                    x,y,fmt = point
                    nulow = x - y/2.0
                    nuhigh = x + y/2.0

                    if self.log:
                        ax.plot(np.log10(x),np.log10(y),fmt,zorder=50,ms=8)
                        nus = np.logspace(np.log10(nulow),np.log10(nuhigh),self.nchan+1)[:-1] 
                        sigma = np.log10(self.calc_single(nus)[0])
                    else:
                        ax.plot(x,y,fmt,zorder=50,ms=8)
                        nus = np.linspace(nulow,nuhigh,self.nchan+1)[:-1] #more uniform sampling?
                        sigma = np.log10(self.calc_single(nus)[0])
                    with open("minima.txt",'a') as FILE:
                        FILE.write("%s point %f %f %f\n"%(self.psrnoise.name,x,y,sigma))




            if colorbararrow is not None:
                data = np.log10(self.sigmas)
                flatdata = data.flatten()
                #inds = np.where(np.logical_not(np.isnan(flatdata)))[0]
                inds = np.where((~np.isnan(flatdata))&~(np.isinf(flatdata)))[0]
                MIN = np.min(flatdata[inds])
                MAX = np.max(flatdata[inds])
                if self.log == True:
                    x = np.log10(self.Cs[-1]*1.05)#self.Bs[-1])
                    dx = np.log10(1.2)#np.log10(self.Cs[-1])#self.Bs[-1]*2)
                    frac = (np.log10(colorbararrow)-MIN)/(MAX-MIN)
                    y = frac*(np.log10(self.Bs[-1]) - np.log10(self.Bs[0])) + np.log10(self.Bs[0])
                    arrow(x,y,dx,0.0,fc='k',ec='k',zorder=50,clip_on=False)




        else:
            if self.log == False:
                pass
            else:
                goodinds = []
                for indf,F in enumerate(self.Fs):
                    if np.any(np.isnan(self.sigmas[:,indf])):
                        continue
                    goodinds.append(indf)
                goodinds = np.array(goodinds)
                data = np.transpose(np.log10(self.sigmas[:,goodinds]))

                im = uimshow(data,extent=np.log10(np.array([self.Cs[0],self.Cs[-1],self.Fs[goodinds][0],self.Fs[goodinds][-1]])),cmap=cmap,ax=ax,**kwargs)
                cax = ax.contour(data,extent=np.log10(np.array([self.Cs[0],self.Cs[-1],self.Fs[goodinds][0],self.Fs[goodinds][-1]])),colors=COLORS,levels=LEVELS,linewidths=LWS,origin='lower')

                #im = uimshow(data,extent=np.array([np.log10(self.Cs[0]),np.log10(self.Cs[-1]),self.Fs[goodinds][0],self.Fs[goodinds][-1]]),cmap=cm.inferno_r,ax=ax)
                #cax = ax.contour(data,extent=np.array([np.log10(self.Cs[0]),np.log10(self.Cs[-1]),self.Fs[goodinds][0],self.Fs[goodinds][-1]]),colors=COLORS,levels=LEVELS,linewidths=LWS,origin='lower')


                
                print(self.Fs)
                ax.set_xlabel(r"$\mathrm{Center~Frequency~\nu_0~(GHz)}$")
                #ax.set_ylabel(r"$r~\mathrm{(\nu_{max}/\nu_{min})}$")
                ax.set_ylabel(r"$\mathrm{Fractional~Bandwidth~(B/\nu_0)}$")
                # no log
                #ax.yaxis.set_major_locator(FixedLocator(np.log10(np.arange(0.25,1.75,0.25))))
                
                ax.xaxis.set_major_formatter(noformatter)
                #ax.yaxis.set_major_formatter(noformatter)
            
            
        cbar = fig.colorbar(im)#,format=formatter)
        cbar.set_label(r"$\mathrm{TOA~Uncertainty~\sigma_{TOA}~(\mu s)}$")

        # https://stackoverflow.com/questions/6485000/python-matplotlib-colorbar-setting-tick-formator-locator-changes-tick-labels
        cbar.locator = MultipleLocator(1)
        cbar.formatter = formatter
        '''
        MAX = np.max(data[np.where(np.logical_not(np.isnan(data)))])
        if MAX <= np.log10(700):
            cbar.formatter = formatter100
        else:
            cbar.formatter = formatter
        '''
        cbar.update_ticks()
        #if self.log:
        #    cb = colorbar(cax)



        if save:
            savefig(filename)
        if doshow:
            show()
        else:
            close()

    def save(self,filename):
        '''
        Output the results of the grid runs to a file
        '''
        if self.frac_bw == False:
            np.savez(filename,Cs=self.Cs,Bs=self.Bs,sigmas=self.sigmas)
        else:
            np.savez(filename,Cs=self.Cs,Fs=self.Fs,sigmas=self.sigmas)

    def get_optimum(self):
        checkdata = np.log10(self.sigmas)
        flatdata = checkdata.flatten()
        #inds = np.where(np.logical_not(np.isnan(flatdata)))[0]
        inds = np.where((~np.isnan(flatdata))&~(np.isinf(flatdata)))[0]
        MIN = np.min(flatdata[inds])
        INDC,INDB = np.where(checkdata==MIN)
        INDC,INDB = INDC[0],INDB[0]
        MINB = self.Bs[INDB]
        MINC = self.Cs[INDC]

        return MINC,MINB
 
