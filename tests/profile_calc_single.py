"""
Run profiler on frequencyoptimizer.FrequencyOptimizer.calc_single.
Execute after adding @profile decorator to function(s).
"""
import re
import sys
import numpy as np
from subprocess import Popen, PIPE
import frequencyoptimizer as fop

def setup():
    """ Set up instance of FrequencyOptimizer"""
    bw = 0.6 # GHz
    n_channels = 100
    ctrfreq = 1.4
    #        nus = np.array([1.1, 1.16, 1.4])
    #        ctrfreq = np.median(nus)
    nus = np.linspace(ctrfreq - bw / 2, ctrfreq + bw / 2, n_channels + 1)[:-1]
    psr_noise = fop.PulsarNoise('',
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
    scope_noise = fop.TelescopeNoise(2.,
                                     T_const=22.73,
                                     T=1800,
                                     epsilon=0.01)
    gal_noise = fop.GalacticNoise()
    fop_inst = fop.FrequencyOptimizer(psr_noise,
                                      gal_noise,
                                      scope_noise,
                                      nchan=len(nus),
                                      numax=ctrfreq,
                                      numin=ctrfreq,
                                      vverbose=False)
    return fop_inst, nus

def main():
    fop_inst, nus = setup()
    fop_inst.calc_single(nus)
        

if __name__ == '__main__':
    sys.exit(main())
