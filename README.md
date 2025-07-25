FrequencyOptimizer
=======

A python package for the optimal frequencies analysis (Lam et al., 2018) of pulsars

Requires:
* python 2.7
* numpy
* scipy
* matplotlib

What's New?
-----------
* In Lam et al. (2018), Equation 4 contains the white noise uncertainty added in quadrature. We have determined that since the uncertainty in DM fit (`sigmadm2`) includes the white noise covariance matrix, that term is extraneous and is no longer included in the total TOA uncertainty returned by `FrequencyOptimizer.calc_single`. We still report the white noise uncertainty separately when `vverbose=True`.
* Support for frequency-dependent sky temperatures, receiver temperatures (`T_const` is now `T_rx`), gain, and fractional gain error
* Default and user-defined receiver specification files
* Speed-ups and bug fixes

PulsarNoise
-----------

A class for describing pulsar noise parameters

Usage: 

    pn = PulsarNoise(name,alpha=1.6,dtd=None,dnud=None,taud=None,C1=1.16,I_0=18.0,DM=0.0,D=1.0,tauvar=None,Weffs=None,W50s=None,sigma_Js=0.0,P=None,glon=None,glat=None)

* alpha: Pulsar flux spectral index
* dtd: Scintillation timescale (s)
* dnud: Scintillation bandwidth (GHz)
* taud: Scattering timescale (us)
* C1: Coefficient relating dnud to taud (1.16 = uniform Kolmogorov medium) 
* I_0: Pulsar flux density at 1 GHz
* DM: Dispersion measure (pc cm^-3)
* D: Distance (kpc)
* Uscale: Dimensionless factor that describes how intensity is distributed across pulse phase, see Sec. 2.2.1 of (Lam et al. 2018)
* tauvar: Variation in scattering timescale (us)
* Weffs: Effective width, can be an array (us)
* W50s: Pulse full-width at half-maximum, can be an array (us)
* sigma_Js: Jitter for observation time T, can be an array (us) [note: T needs to be related to the TelescopeNoise class]
* glon: Galactic longitude (deg)
* glat: galactic latitude (deg)


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
       
    tn = TelescopeNoise(gain,T_rx,epsilon=0.08,pi_V=0.1,eta=0.0,pi_L=0.0,T=1800.0,Npol=2,rx_nu=None,interpolate=False)

* gain: Telescope gain (K/Jy), if array must be same length as rx_nu 
* T_rx: Receiver temperature (K) (i.e. T_sys - T_gal - T_CMB), if array must be same length as rx_nu 
* epsilon: Fractional gain error, if array must be same length as rx_nu
* pi_V: Degree of circular polarization
* eta: Voltage cross-coupling coefficient
* pi_L: Degree of linear polarization
* T: Integration time (s)
* Npol: Number of polarization states
* rx_nu: Receiver frequencies over which to interpolate (GHz)
* interpolate: (boolean) must be set to True to interpolate gain, T_rx, and/or eps
* rxspecfile : string (optional)
  Name of receiver specifications file or path to user-defined file. A user-defined file takes precedence over default files. I.e. A file in the working directory will override a default file with the same name. Call frequencyoptimizer.get_rxspecs_options() to see default files.
  If defined, a file overrides gain, T_rx, and epsilon arguments. Files must contain a header with the format
```
                #Freq  Trx  G  Eps
```
immediately followed by 4 tab-separated columns of frequency, T_rx, gain, and epsilon.

FrequencyOptimizer
------------------

A class for handling calculations

Usage: 

    freqopt = FrequencyOptimizer(psrnoise,galnoise,telnoise,numin=0.01,numax=10.0,dnu=0.05,nchan=100,log=False,nsteps=8,frac_bw=False,verbose=True,full_bandwidth=False,masks=None,levels=LEVELS,colors=COLORS,lws=LWS, ncpu=1)
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
* ncpu: number of cores for multiprocess threading

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
    telnoise = TelescopeNoise(gain=2.0,T_rx=30)

    NCHAN = 100
    NSTEPS = 100
    psrnoise = PulsarNoise("J1744-1134",alpha=1.49,taud=26.1e-3,I_0=4.888,DM=3.14,D=0.41,tauvar=12.2e-3,dtd=1272.2,Weffs=np.zeros(NCHAN)+511.0,W50s=np.zeros(NCHAN)+136.8,sigma_Js=np.zeros(NCHAN)+0.066,P=4.074545941439190)  #jitter upper limit

    freqopt = FrequencyOptimizer(psrnoise,galnoise,telnoise,numin=0.1,numax=10.0,nchan=NCHAN,log=True,nsteps=NSTEPS)
    freqopt.calc()
    freqopt.plot("J1744-1134.png",doshow=False,minimum='k*',points=(1.3,1.2,'ko'))
    freqopt.save("J1744-1134.npz")





Citations
---------

If you use FrequencyOptimizer in work that results in a publication, please use the following attribution:

```
@ARTICLE{Lam2018FrequencyOptimizer,
       author = {{Lam}, M.~T. and {McLaughlin}, M.~A. and {Cordes}, J.~M. and {Chatterjee}, S. and {Lazio}, T.~J.~W.},
        title = "{Optimal Frequency Ranges for Submicrosecond Precision Pulsar Timing}",
      journal = {\apj},
     keywords = {gravitational waves, methods: observational, pulsars: general, Astrophysics - High Energy Astrophysical Phenomena, Astrophysics - Instrumentation and Methods for Astrophysics},
         year = 2018,
        month = jul,
       volume = {861},
       number = {1},
          eid = {12},
        pages = {12},
          doi = {10.3847/1538-4357/aac48d},
archivePrefix = {arXiv},
       eprint = {1710.02272},
 primaryClass = {astro-ph.HE},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2018ApJ...861...12L},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
```
