import numpy as np
import frequencyoptimizer as fop

def test1():
    TCMB = 2.73
    ctrfreq = 1.4 #GHz
    t_int = 1800 # seconds
    nus = np.array([1.1, 1.16, 1.22, 1.28, 1.34, 1.4, 1.46, 1.52, 1.58, 1.64])

    pulsar_noise = fop.PulsarNoise('',
                                   alpha=1.28,
                                   dtd=0.048,
                                   dnud=1.,
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
                                     T_const=20 + TCMB,
                                     T=t_int,
                                     epsilon=0.01)
    gal_noise = fop.GalacticNoise()
    fop_inst = fop.FrequencyOptimizer(pulsar_noise,
                                      gal_noise,
                                      scope_noise,
                                      nchan=len(nus),
                                      numax=ctrfreq,
                                      numin=ctrfreq,
                                      vverbose=False)

    print(fop_inst.build_DMnu_cov_matrix(nus))

    return
