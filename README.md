Note: Original code by Michael T. Lam
https://github.com/mtlam/FrequencyOptimizer
Modified by Ryan S. Lynch

FrequencyOptimizer
=======

A python package for the optimal frequencies analysis (M. T. Lam et al in prep.) of pulsars

Requires:
python 2.7
numpy
scipy
matplotlib


PulsarNoise
-----------

A class for describing pulsar noise parameters

Usage: 

    pn = PulsarNoise(name,alpha=1.6,dtd=None,dnud=None,taud=None,C1=1.16,I_0=18.0,DM=0.0,D=1.0,tauvar=None,Weffs=None,W50s=None,sigma_Js=None,P=None)

* alpha: Pulsar flux spectral index
* dtd: Scintillation timescale (s)
* dnud: Scintillation bandwidth (GHz)
* taud: Scattering timescale (us)
* C1: Coefficient relating dnud to taud (1.16 = uniform Kolmogorov medium) 
* I_0: Pulsar flux density at 1 GHz
* DM: Dispersion measure (pc cm^-3)
* D: Distance (kpc)
* tauvar: Variation in scattering timescale (us)
* Weffs: Effective width, can be an array (us)
* W50s: Pulse full-width at half-maximum, can be an array (us)
* sigma_Js: Jitter for observation time T, can be an array (us) [note: T needs to be related to the TelescopeNoise class]


GalacticNoise
-------------

A class for describing galaxy parameters

Usage: 

    gn = GalacticNoise(beta=2.75,T_e=100.0,fillingfactor=0.2)

* beta: Galactic sky background spectral index
* T_e: electron temperature (for emission measure analysis, now defunct)
* fillingfactor: filling factor (for emission measure analysis, now defunct)

TelescopeNoise
--------------

A class for describing telescope noise parameters

Usage: 
       
    tn = TelescopeNoise(gain,T_const,epsilon=0.08,pi_V=0.1,eta=0.0,pi_L=0.0,T=1800.0)

* gain: Telescope gain (K/Jy)
* T_const: Constant temperature (e.g. T_sys + T_CMB + ...)
* epsilon: Fractional gain error
* pi_V: Degree of circular polarization
* eta: Voltage cross-coupling coefficient
* pi_L: Degree of linear polarization
* T: Integration time (s)


FrequencyOptimizer
------------------

A class for handling calculations

Usage: 

    freqopt = FrequencyOptimizer(psrnoise,galnoise,telnoise,numin=0.01,numax=10.0,dnu=0.05,nchan=100,log=False,nsteps=8,frac_bw=False,verbose=True,full_bandwidth=False,masks=None,levels=LEVELS,colors=COLORS,lws=LWS)
    freqopt.calc() #calculate
    freqopt.plot(filename="triplot.png",doshow=True,figsize=(8,6),save=True,minimum=None,points=None,colorbararrow=None) #plot/save figure
    freqopt.save(filename) #save to .npz file

* psrnoise: Pulsar Noise object
* galnoise: Galaxy Noise object
* telnoise: Telescope Noise object
* numin: Lowest frequency to run (GHz)
* numax: Highest frequency to run (GHz)
* nsteps: Number of steps in the grid to run
* nchan: number of underlying frequency channels
* log: Run in log space
* frac_bw: Run in fractional bandwidth
* full_bandiwdth: enforce full bandwidth in calculations
* masks: mask frequencies [not fully implemented]
* levels: contour levels
* colors: contour colors
* lws: contour linewidths

For plotting:

* filename: filename
* doshow: show the plot
* figsize = figure size
* save: Save the figure
* minimum: Symbol to place over the minimum
* points: Place other points on the plot



Sample Code
-----------

    galnoise = GalacticNoise()
    telnoise = TelescopeNoise(gain=2.0,T_const=30)

    NCHAN = 100
    NSTEPS = 100
    psrnoise = PulsarNoise("J1744-1134",alpha=1.49,taud=26.1e-3,I_0=4.888,DM=3.14,D=0.41,tauvar=12.2e-3,dtd=1272.2,Weffs=np.zeros(NCHAN)+511.0,W50s=np.zeros(NCHAN)+136.8,sigma_Js=np.zeros(NCHAN)+0.066,P=4.074545941439190)  #jitter upper limit

    freqopt = FrequencyOptimizer(psrnoise,galnoise,telnoise,numin=0.1,numax=10.0,nchan=NCHAN,log=True,nsteps=NSTEPS)
    freqopt.calc()
    freqopt.plot("J1744-1134.png",doshow=False,minimum='k*',points=(1.3,1.2,'ko'))
    freqopt.save("J1744-1134.npz")





Citations
---------

Please cite this github page currently.