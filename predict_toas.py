import numpy as np
import psr_utils as pu
import matplotlib.pyplot as plt
import pyslalib.slalib as slalib
from argparse import ArgumentParser
from frequencyoptimizer import PulsarNoise,TelescopeNoise,GalacticNoise
from frequencyoptimizer import FrequencyOptimizer

NCHAN = 16
NSTEPS = 16

Tcmb = 3.0

parser = ArgumentParser(description="Predict pulsar TOA precision")
rx_group = parser.add_argument_group(title="Receiver specs")
pulsar_group = parser.add_argument_group(title="Pulsar parameters")
rx_group.add_argument("-r", "--rx-specs", 
                      help="File containing receiver performance specs")
rx_group.add_argument("-L", "--low-freq", type=float, default=1.0,
                      help="Low frequency (GHz; default=%(default)s)")
rx_group.add_argument("-H", "--high-freq", type=float, default=2.0,
                      help="High frequency (GHz; default=%(default)s)")
rx_group.add_argument("-T", "--trx", dest="Trx", type=float, default=20, 
                      help="Receiver temperature (K; default=%(default)s)")
rx_group.add_argument("-G", "--gain", type=float, default=2.0, 
                      help="Telescope gain (K/Jy; default=%(default)s)")
rx_group.add_argument("-e", "--epsilon", type=float, default=0.01,
                      help="Fractional gain instability (default=%(default)s)")
rx_group.add_argument("-p", "--pi", dest="pi_V", type=float, default=0.95,
                      help="Polarization efficiency (default=%(default)s)")
parser.add_argument("-t", "--tobs", type=float, default=1800.0, 
                    help="Observing time (s; default=%(default)s)")

pulsar_group.add_argument("-d", "--psr-dict", 
                          help="Python dictionary containing pulsar parameters")

pulsar_group.add_argument("-n", "--name", default="Fake",
                          help="Pulsar name (default=%(default)s)")
pulsar_group.add_argument("-P", "--period", type=float, default=3.0,
                          help="Pulsar period (ms; default=%(default)s)")
pulsar_group.add_argument("-D", "--dm", type=float, default=30.0,
                          help="Pulsar DM (pc/cc; default=%(default)s)")
pulsar_group.add_argument("-F", "--flux", dest="flux_1GHz", type=float,
                          default=10.0,
                          help=("Pulsar flux density at 1 GHz (mJy; "
                                "default=%(default)s)"))
pulsar_group.add_argument("-a", "--alpha", dest="spec_index", type=float,
                          default=-1.7,
                          help="Pulsar spectral index (default=%(default)s)")
pulsar_group.add_argument("-w", "--width", dest="W50", type=float,
                          default=300.0,
                          help="Pulsar FWHM (us; default=%(default)s)")
pulsar_group.add_argument("--weff", dest="Weff", type=float,
                          help="Pulsar effective width (us; default=1.2 x W50)")
pulsar_group.add_argument("-U", "--uscale", type=float, default=10.0, 
                          help=("Pulsar profile scaling factor"))
pulsar_group.add_argument("-j", "--jitter", dest="rms_J1", type=float,
                          default=100.0, 
                          help=("Pulsar single-pulse jitter RMS (us; "
                                "default=%(default)s)"))
pulsar_group.add_argument("-s", "--scat-ts", type=float, default=0.01, 
                          help=("Pulsar scattering timescale at 1 GHz "
                                "(us; default=%(default)s)"))
pulsar_group.add_argument("-S", "--scat-ts-var", type=float, default=0.05, 
                          help=("Pulsar scattering timescale variability at "
                                "1 GHz (us; default=%(default)s)"))
pulsar_group.add_argument("-i", "--diss-ts", type=float, default=1000.0, 
                          help=("Pulsar diffractive ISS timescale at 1 GHz (s; "
                                "default=%(default)s"))
args = parser.parse_args()

if args.rx_specs is not None:
    interpolate = True
    rx_freq,Trx,gain,epsilon,pi_V = np.loadtxt(args.rx_specs,unpack=True)
    low_freq = np.min(rx_freq)
    high_freq = np.max(rx_freq)
    freqs = np.logspace(np.log10(low_freq),np.log10(high_freq),NCHAN)
else:
    interpolate = False
    Trx = args.Trx
    gain = args.gain
    epsilon = args.epsilon
    pi_V = args.pi_V
    low_freq = args.low_freq
    high_freq = args.high_freq
    freqs = np.logspace(np.log10(low_freq),np.log10(high_freq),NCHAN)
    rx_freq = freqs.copy()

if args.psr_dict is not None:
    import importlib
    try:
        psr_dict = importlib.import_module(args.psr_dict.rstrip(".py"))
        for a in dir(psr_dict):
            if not a.startswith("__"):
                psrs = psr_dict.__getattribute__(a)
                break
    except ImportError:
        with open(args.psr_dict) as f:
            lines = [line.split() for line in f]
            psrs = {}
            for line in lines:
                d = {line[0]: line[1] for line in lines}
                psrs[d["name"]] = d
else:
    psrs = {args.name: {
        "name": args.name,
        "period": args.period,
        "DM": args.dm,
        "flux_1GHz": args.flux_1GHz,
        "spec_index": args.spec_index,
        "W50": args.W50,
        "Weff": 1.2*args.W50 if args.Weff is None else args.Weff,
        "uscale": args.uscale,
        "rms_J1": args.rms_J1,
        "scat_ts": args.scat_ts,
        "scat_ts_var": args.scat_ts_var,
        "diss_ts": args.diss_ts }}

sigmas = []
telescope_noise = TelescopeNoise(gain=gain,T_const=Trx+Tcmb,epsilon=epsilon,
                                 pi_V=pi_V,T=args.tobs,rx_nu=rx_freq,
                                 interpolate=interpolate)
galactic_noise = GalacticNoise()
for name,psr in psrs.items():
    if psr["W50"] is None and psr["Weff"] is not None: 
        psr["W50"] = 0.66*psr["Weff"]
    if psr["scat_ts"] is None and psr["DM"] is not None:
        psr["scat_ts"] = 1000*pu.pulse_broadening(psr["DM"],1000.0)
    if None not in psr.values():
        pulsar_noise = PulsarNoise(
            name,alpha=-psr["spec_index"],I_0=psr["flux_1GHz"],DM=psr["DM"],
            taud=psr["scat_ts"]*1.5**4.4,tauvar=psr["scat_ts_var"]*1.5**4.4*0.5,
            dtd=psr["diss_ts"],P=psr["period"],Uscale=psr["uscale"],
            sigma_Js=np.zeros(NCHAN)+psr["rms_J1"] / \
            np.sqrt(1000*args.tobs/psr["period"]),
            Weffs=np.zeros(NCHAN)+psr["Weff"],W50s=np.zeros(NCHAN)+psr["W50"])
        frequency_optimizer = FrequencyOptimizer(
            pulsar_noise,galactic_noise,telescope_noise,
            numin=low_freq,numax=high_freq,nchan=NCHAN,log=True,nsteps=NSTEPS,
            frac_bw=False,full_bandwidth=False,masks=None)
        sigma = frequency_optimizer.calc_single(freqs)
        sigmas.append(sigma)
        print("%-10s   %.3f"%(name,sigma))

sigma_mean = np.mean(sigmas)
sigma_median = np.median(sigmas)
sigma_std = np.std(sigmas)

print("Mean sigma = %.3f"%sigma_mean)
print("Median sigma = %.3f"%sigma_median)
print("STD sigma = %.3f"%sigma_std)
