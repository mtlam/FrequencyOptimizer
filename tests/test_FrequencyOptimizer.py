"""
Integration tests for frequencyoptimizer.FrequencyOptimizer
"""

import pytest
import numpy as np
import frequencyoptimizer as fop

@pytest.mark.parametrize("vverbose", [True, False])
def test_FrequencyOptimizer_calc_single_vverbose(vverbose):
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
                               Weffs=np.zeros(nchan)+511.0,
                               W50s=np.zeros(nchan)+136.8,
                               sigma_Js=np.zeros(nchan)+0.066,
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
