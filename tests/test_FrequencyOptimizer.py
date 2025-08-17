"""
Integration tests for frequencyoptimizer.FrequencyOptimizer
"""

import pytest
import multiprocessing as mpc
import numpy as np
import frequencyoptimizer as fop
from matplotlib import cm

PSR_PARAMS = {'J1713+0747AO': {'alpha': 1.2,
                               'taud': 0.0521,
                               'I_0': 10.323,
                               'DM': 15.97,
                               'D': 1.18,
                               'tauvar': 0.0354,
                               'dtd': 1755.0,
                               'Weffs': 539.0,
                               'W50s': 110.4,
                               'sigma_Js': 0.039,
                               'P': 4.570136526356975,
                               'Uscale': 26.41},
              'J1713+0747GBT': {'alpha': 1.2,
                                'taud': 0.0521,
                                'I_0': 10.323,
                                'DM': 15.97,
                                'D': 1.18,
                                'tauvar': 0.0354,
                                'dtd': 1755.0,
                                'Weffs': 533.0,
                                'W50s': 108.5,
                                'sigma_Js': 0.051,
                                'P': 4.570136526356975,
                                'Uscale': 26.83},
              'J19093744': {'alpha': 1.89,
                            'taud': 0.0282,
                            'I_0': 2.64,
                            'DM': 10.39,
                            'D': 1.06,
                            'tauvar': 0.0208,
                            'dtd': 1386.1,
                            'Weffs': 261.0,
                            'W50s': 41.0,
                            'sigma_Js': 0.014,
                            'P': 2.947108069160717,
                            'Uscale': 61.37},
              'J1903+0327': {'alpha': 2.08,
                             'taud': 554.0,
                             'I_0': 2.614,
                             'DM': 297.52,
                             'D': 2.5,
                             'tauvar': None,
                             'dtd': 7.405,
                             'Weffs': 405.0,
                             'W50s': 195.4,
                             'sigma_Js': 0.257,
                             'P': 2.14991236434921,
                             'Uscale': 9.28},
              'J17441134': {'alpha': 1.49,
                            'taud': 0.0261,
                            'I_0': 4.888,
                            'DM': 3.14,
                            'D': 0.41,
                            'tauvar': 0.0122,
                            'dtd': 1272.2,
                            'Weffs': 511.0,
                            'W50s': 136.8,
                            'sigma_Js': 0.066,
                            'P': 4.07454594143919,
                            'Uscale': 27.01},
              'J16431224': {'alpha': 2.23,
                            'taud': 43.0,
                            'I_0': 9.127,
                            'DM': 62.41,
                            'D': 1.43,
                            'tauvar': 0.1 * 43.0,
                            'dtd': 357.9,
                            'Weffs': 973.0,
                            'W50s': 314.6,
                            'sigma_Js': 0.219,
                            'P': 4.62164152493627,
                            'Uscale': 10.22}}

@pytest.mark.parametrize("vverbose", [True, False])
def test_FrequencyOptimizer_calc_single_vverbose(vverbose):
    """
    Test frequencyoptimizer.FrequencyOptimizer.calc_single with
    and without printing all noise components to stdout
    """
    nus = np.array([0.72370136, 0.73018196, 0.73672059, 0.74331777,
                    0.74997403, 0.75668989, 0.7634659, 0.77030258,
                    0.77720048, 0.78416015, 0.79118215, 0.79826702,
                    0.80541534, 0.81262767, 0.81990458, 0.82724666,
                    0.83465449, 0.84212865, 0.84966974, 0.85727836])
    NCHAN = len(nus)
    galnoise = fop.GalacticNoise()
    telnoise = fop.TelescopeNoise(gain=2.0, T_rx=30.)

    psrnoise = fop.PulsarNoise("J1744-1134",
                               alpha=1.49,
                               taud=26.1e-3,
                               I_0=4.888,
                               DM=3.14,
                               D=0.41,
                               tauvar=12.2e-3,
                               dtd=1272.2,
                               Weffs=np.zeros(NCHAN)+511.0,
                               W50s=np.zeros(NCHAN)+136.8,
                               sigma_Js=np.zeros(NCHAN)+0.066,
                               P=4.074545941439190,
                               Uscale=27.01)

    freqopt = fop.FrequencyOptimizer(psrnoise,
                                     galnoise,
                                     telnoise,
                                     numin=0.1,
                                     numax=10.0,
                                     nchan=NCHAN,
                                     log=True,
                                     vverbose=vverbose)

    freqopt.calc_single(nus)

@pytest.mark.parametrize("vverbose", [True, False])
def test_FrequencyOptimizer_calc_vverbose(vverbose):
    """
    Test frequencyoptimizer.FrequencyOptimizer.calc_single with
    and without printing all noise components to stdout
    """
    nchan = 20
    galnoise = fop.GalacticNoise()
    telnoise = fop.TelescopeNoise(gain=2.0, T_rx=30.)

    psrnoise = fop.PulsarNoise("J1744-1134",
                               alpha=1.49,
                               taud=26.1e-3,
                               I_0=4.888,
                               DM=3.14,
                               D=0.41,
                               tauvar=12.2e-3,
                               dtd=1272.2,
                               Weffs=np.full(nchan, 511.0),
                               W50s=np.full(nchan, 136.80),
                               sigma_Js=np.full(nchan, 0.066),
                               P=4.074545941439190,
                               Uscale=27.01)

    freqopt = fop.FrequencyOptimizer(psrnoise,
                                     galnoise,
                                     telnoise,
                                     numin=0.1,
                                     numax=10.0,
                                     nchan=nchan,
                                     log=True,
                                     vverbose=vverbose)

    freqopt.calc()

@pytest.mark.parametrize("psr", list(PSR_PARAMS))
def test_FrequencyOptimizer_plot_pulsarparams(psr):
    """
    Test plot optimal center frequency/bandwidth for different pulsars
    """
    nchan = 20
    galnoise = fop.GalacticNoise()
    telnoise = fop.TelescopeNoise(gain=2.,
                                  T_rx=30.)

    psrnoise = fop.PulsarNoise(psr,
                               alpha=PSR_PARAMS[psr]["alpha"],
                               taud=PSR_PARAMS[psr]["taud"],
                               I_0=PSR_PARAMS[psr]["I_0"],
                               DM=PSR_PARAMS[psr]["DM"],
                               D=PSR_PARAMS[psr]["D"],
                               tauvar=PSR_PARAMS[psr]["tauvar"],
                               dtd=PSR_PARAMS[psr]["dtd"],
                               Weffs=np.full(nchan, PSR_PARAMS[psr]["Weffs"]),
                               W50s=np.full(nchan, PSR_PARAMS[psr]["W50s"]),
                               sigma_Js=np.full(nchan, PSR_PARAMS[psr]["sigma_Js"]),
                               P=PSR_PARAMS[psr]["P"],
                               Uscale=PSR_PARAMS[psr]["Uscale"])

    freqopt = fop.FrequencyOptimizer(psrnoise,
                                     galnoise,
                                     telnoise,
                                     numin=0.1,
                                     numax=10.0,
                                     nsteps=25,
                                     nchan=nchan,
                                     log=True,
                                     full_bandwidth=True,
                                     levels=np.array([np.log10(0.1),
                                                      np.log10(0.2),
                                                      np.log10(0.5),
                                                      np.log10(1.0),
                                                      np.log10(2.0)]),
                                     vverbose=False)
    freqopt.calc()

    freqopt.plot(save=False,
                 doshow=False,
                 minimum="k*",
                 points=[(1.8, 1.2,"ko")],
                 colorbararrow=None,
                 cmap=cm.inferno_r)
    

@pytest.mark.parametrize("psr", list(PSR_PARAMS))
def test_FrequencyOptimizer_calc_pulsarparams(psr):
    """
    Test optimization for different pulsars from Michael's paper
    """
    nchan = 20
    galnoise = fop.GalacticNoise()
    telnoise = fop.TelescopeNoise(gain=2.,
                                  T_rx=30.)

    psrnoise = fop.PulsarNoise(psr,
                               alpha=PSR_PARAMS[psr]["alpha"],
                               taud=PSR_PARAMS[psr]["taud"],
                               I_0=PSR_PARAMS[psr]["I_0"],
                               DM=PSR_PARAMS[psr]["DM"],
                               D=PSR_PARAMS[psr]["D"],
                               tauvar=PSR_PARAMS[psr]["tauvar"],
                               dtd=PSR_PARAMS[psr]["dtd"],
                               Weffs=np.full(nchan, PSR_PARAMS[psr]["Weffs"]),
                               W50s=np.full(nchan, PSR_PARAMS[psr]["W50s"]),
                               sigma_Js=np.full(nchan, PSR_PARAMS[psr]["sigma_Js"]),
                               P=PSR_PARAMS[psr]["P"],
                               Uscale=PSR_PARAMS[psr]["Uscale"])

    freqopt = fop.FrequencyOptimizer(psrnoise,
                                     galnoise,
                                     telnoise,
                                     numin=0.1,
                                     numax=10.0,
                                     nsteps=25,
                                     nchan=nchan,
                                     log=True,
                                     full_bandwidth=True,
                                     vverbose=False)
    freqopt.calc()
    
@pytest.mark.parametrize("ncpus", np.arange(mpc.cpu_count() - 2) + 1)
def test_FrequencyOptimizer_calc_ncpus(ncpus):
    """
    Integration test for parallel.py with
    frequencyoptimizer.FrequencyOptimizer.calc. No multiprocessing 
    used when ncpus=1
    """
    nchan = 20
    galnoise = fop.GalacticNoise()
    telnoise = fop.TelescopeNoise(gain=2.0, T_rx=30.)

    psrnoise = fop.PulsarNoise("J1744-1134",
                               alpha=1.49,
                               taud=26.1e-3,
                               I_0=4.888,
                               DM=3.14,
                               D=0.41,
                               tauvar=12.2e-3,
                               dtd=1272.2,
                               Weffs=np.full(nchan, 511.0),
                               W50s=np.full(nchan, 136.8),
                               sigma_Js=np.full(nchan, 0.066),
                               P=4.074545941439190,
                               Uscale=27.01)

    freqopt = fop.FrequencyOptimizer(psrnoise,
                                     galnoise,
                                     telnoise,
                                     numin=0.1,
                                     numax=10.0,
                                     nchan=nchan,
                                     ncpu=ncpus,
                                     log=True,
                                     vverbose=False)

    freqopt.calc()

@pytest.mark.parametrize(("log", "dnu", "full_bw", "r"),
                         [(True, None, False, None),
                          (False, 0.05, False, None),
                          (True, None, False, 2)])
def test_FrequencyOptimizer_calc_NaN_conditions(log, dnu, full_bw, r):
    """
    Check that sigmas have np.nan in the right location based on
    conditions
    """
    nchan = 20
    nsteps = 20
    galnoise = fop.GalacticNoise()
    telnoise = fop.TelescopeNoise(gain=2.0, T_rx=30.)
    
    psrnoise = fop.PulsarNoise("J1744-1134",
                               alpha=1.49,
                               taud=26.1e-3,
                               I_0=4.888,
                               DM=3.14,
                               D=0.41,
                               tauvar=12.2e-3,
                               dtd=1272.2,
                               Weffs=np.full(nchan, 511.0),
                               W50s=np.full(nchan, 136.8),
                               sigma_Js=np.full(nchan, 0.066),
                               P=4.074545941439190,
                               Uscale=27.01)

    freqopt = fop.FrequencyOptimizer(psrnoise,
                                     galnoise,
                                     telnoise,
                                     numin=0.1,
                                     numax=10.0,
                                     nchan=nchan,
                                     ncpu=1,
                                     nsteps=nsteps,
                                     log=log,
                                     dnu=dnu,
                                     full_bandwidth=full_bw,
                                     r=r,
                                     verbose=False)
    freqopt.calc()

    # correct NaN location mask
    if log == False:
       dnu = dnu
       Cs = np.arange(freqopt.numin, freqopt.numax, dnu)
       Bs = np.arange(freqopt.numin, freqopt.numax / 2, dnu)
    else:
       MIN = np.log10(freqopt.numin)
       MAX = np.log10(freqopt.numax)
       Cs = np.logspace(MIN, MAX, int((MAX - MIN) * nsteps + 1))
       if full_bw:
           MAX = np.log10(2 * numax)
           Bs = np.logspace(MIN, MAX, int((MAX - MIN) * nsteps + 1))
       else:
           Bs = np.logspace(MIN, MAX, int((MAX - MIN) * nsteps + 1))

    C, B = np.meshgrid(Cs, Bs)
    cond1 = B > 1.9 * C
    cond2 = (C - B / 2.0) < freqopt.numin # nearly redundant to cond1
    if r is not None:
        cond3 = (C + 0.5 * B) / (C - 0.5 * B) > r
    else:
        cond3 = np.full((len(Bs), len(Cs)), False)
        
    NaN_location = cond1 | cond2 | cond3
    
    np.testing.assert_equal(NaN_location, np.isnan(freqopt.sigmas).T)
