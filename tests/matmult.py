import cPickle
from os import path
import numpy as np
import frequencyoptimizer as fop
import line_profiler

# Load population from file
popdir = '/users/tcohen/Documents/Research_Journal/MSP_Populations/models'
modf = 'AO_430_Lwide_V2.pta.model'
with open(path.join(popdir, modf), 'rb') as modfile:
    pta = cPickle.load(modfile)

for p in pta.population:
    if p.sigma_toa > 0.:
        psrobj = p
        break

# Read in GBT Rcvr1_2 temps and gain
fopdir = path.dirname(path.abspath(fop.__file__))
rxspecdir = path.join(fopdir, 'rxspecs')
rxspecfile = 'GBT_LBand_const.txt'
rx_freqs, Trxs, Gains, Eps = np.loadtxt(path.join(rxspecdir, rxspecfile),
                                        unpack=True)
T_CMB = 2.73

# Observation Parameters
ctrfreq = 1.4 #GHz
bw = 0.6 # GHz
t_int = 1800 # seconds
n_channels = 100
nus = np.linspace(ctrfreq - bw / 2, ctrfreq + bw / 2, n_channels + 1)[:-1]

# Set up FrequencyOptimizer
pulsar_noise = fop.PulsarNoise('', 
                               alpha=psrobj.spindex_neg,
                               dtd=psrobj.dtd(),
                               dnud=None,
                               taud=psrobj.t_scatter_1000(),
                               C1=1.16,
                               I_0=psrobj.s_1000(),
                               DM=psrobj.dm,
                               D=psrobj.dtrue,
                               tauvar=0.5 * psrobj.t_scatter_1000(),
                               Weffs=psrobj.weff(),
                               W50s=psrobj.width_us(),
                               Uscale=psrobj.uscale(),
                               sigma_Js=psrobj.sigmaj(t_int),
                               glon=psrobj.gb,
                               glat=psrobj.gl)
scope_noise = fop.TelescopeNoise(Gains,
                                 T_const=Trxs + T_CMB,
                                 T=t_int,
                                 epsilon=Eps,
                                 rx_nu=rx_freqs,
                                 interpolate=True)
gal_noise = fop.GalacticNoise()
fop_inst = fop.FrequencyOptimizer(pulsar_noise,
                                  gal_noise,
                                  scope_noise,
                                  nchan=len(nus),
                                  numax=ctrfreq,
                                  numin=ctrfreq,
                                  vverbose=False)

# Profile it
fop_inst.calc_single(nus)
# sncov = fop_inst.build_template_fitting_cov_matrix(nus)
# jittercov = fop_inst.build_jitter_cov_matrix() #needs to have same length as nus!
# disscov = fop_inst.build_scintillation_cov_matrix(nus)
# cov = sncov + jittercov + disscov

# profile = line_profiler.LineProfiler()
# profile.add_function(fop.FrequencyOptimizer.DM_misestimation)

# #profile.add_function(fop_inst.DM_misestimation(nus,cov,covmat=True))

# profile.print_stats()
