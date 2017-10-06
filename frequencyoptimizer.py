import numpy as np
import scipy.linalg as linalg
import scipy.interpolate as interpolate
import scipy.optimize as optimize
from matplotlib.pyplot import *
from matplotlib import cm,rc
from matplotlib.ticker import FuncFormatter, MultipleLocator
import matplotlib.patches as patches
import DISS
import glob


rc('text',usetex=True)
rc('font',**{'family':'serif','serif':['Times New Roman'],'size':14})#,'weight':'bold'})
rc('xtick',**{'labelsize':16})
rc('ytick',**{'labelsize':16})
rc('axes',**{'labelsize':18,'titlesize':18})


def nolog(x,pos):
    return "$\hfill %0.1f$" % (10**x)
noformatter = FuncFormatter(nolog)
def nolog2(x,pos):
    return "$\hfill %0.2f$" % (10**x)
noformatter2 = FuncFormatter(nolog2)

def log(x,pos):
    y = x#np.log10(x)
    #if y == 2:
    #    return "$\hfill 100$" #added
    if y == 1:
        return "$\hfill 10$"
    elif y == 0:
        return "$\hfill 1$"
    elif y == -1:
        return "$\hfill 0.1$"
    elif y == -2:
        return "$\hfill 0.01$"
    return "$\hfill 10^{%i}$" % x#np.log10(x) 

formatter = FuncFormatter(log)



def log100(x,pos):
    y = x#np.log10(x)
    if y == 2:
        return "$\hfill 100$" #added
    elif y == 1:
        return "$\hfill 10$"
    elif y == 0:
        return "$\hfill 1$"
    elif y == -1:
        return "$\hfill 0.1$"
    elif y == -2:
        return "$\hfill 0.01$"
    return "$\hfill 10^{%i}$" % x#np.log10(x) 

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
#LEVELS = np.array([np.log10(0.5),np.log10(1.0),np.log10(2.0),np.log10(5.0),np.log10(10.0),np.log10(20.0)])

COLORS = ['k','0.25','0.5','0.75']
COLORS = ['k','0.25','0.5','0.75','1.0']
#COLORS = ['k','0.2','0.4','0.6','0.8','1.0']

LWS = [2.5,2,1.5,1,0.5]
LWS = [2.5,2.25,2,1.75,1.5]
#LWS = [2.5,2.25,2.0,1.75,1.5,1.25]

def epoch_averaged_error(C,var=False):
    # Stripped down version from rednoisemodel.py from the excess noise project
    N = len(C)
    UT = np.matrix(np.ones(N))
    U = UT.T
    CI = C.I
    C_E = np.dot(np.dot(UT,CI),U).I
    if var:
        return C_E[0,0]
    return np.sqrt(C_E[0,0])

def isMatrix(value):
    if type(value) == np.matrixlib.defmatrix.matrix or type(value) == np.ndarray:
        return True
    return False







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
def evalDMnuError(dnuiss,nu1,nu2,g=0.46,q=1.15,screen=False,fresnel=False):
    # nu2 should be less than nu1
    # nu in GHz, dnuiss in GHz
    # return value in microseconds
    # Based on equation 25
    # if fresnel==True, the first argument is phiF
    if screen:
        g = 1
    if fresnel:
        phiF = dnuiss
    else:
        phiF = 9.6 * ((nu1 / dnuiss)/100)**(5.0/12) #equation 15
    r = nu1/nu2
    return 0.184 * g * q * E_beta(r) * (phiF**2 / (nu1 * 1000))





class PulsarNoise:
    def __init__(self,name,alpha=1.7,dtd=None,dnud=None,taud=None,C1=1.16,I_0=18.0,DM=0.0,D=1.0,tauvar=None,Weffs=None,W50s=None,sigma_Js=None,P=None):
        self.name = name

        self.dtd = dtd

        if taud is None:
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

        if P is not None:
            self.P = P * 1000 # now in microseconds
        else:
            self.P = None


class GalacticNoise:
    def __init__(self,beta=2.75,T_e=100.0,fillingfactor=0.2):
        self.beta = beta
        self.T_e = T_e
        self.fillingfactor = fillingfactor


class TelescopeNoise:
    def __init__(self,gain,T_const,epsilon=0.08,pi_V=0.1,eta=0.0,pi_L=0.0,T=1800.0):
        self.gain = gain
        self.T_const = T_const
        self.epsilon = epsilon
        self.pi_V = pi_V
        self.eta = eta
        self.pi_L = pi_L
        self.T = T





class FrequencyOptimizer:
    def __init__(self,psrnoise,galnoise,telnoise,numin=0.01,numax=10.0,dnu=0.05,nchan=100,log=False,nsteps=8,frac_bw=False,verbose=True,full_bandwidth=False,masks=None,levels=LEVELS,colors=COLORS,lws=LWS):



        self.psrnoise = psrnoise
        self.galnoise = galnoise
        self.telnoise = telnoise
        self.log = log
        self.frac_bw = frac_bw
        
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
                self.Cs = np.logspace(MIN,MAX,(MAX-MIN)*nsteps+1)
                if full_bandwidth:
                    MAX = np.log10(2*numax)
                    self.Bs = np.logspace(MIN,MAX,(MAX-MIN)*nsteps+1) 
                else:
                    self.Bs = np.logspace(MIN,MAX,(MAX-MIN)*nsteps+1)
        else:
            if self.log == False:
                pass
            else:
                MIN = np.log10(numin)
                MAX = np.log10(numax)
                self.Cs = np.logspace(MIN,MAX,(MAX-MIN)*nsteps+1)
                self.Bs = np.logspace(MIN,MAX,(MAX-MIN)*nsteps+1)
                self.Fs = np.logspace(np.log10(self.Bs[-1]/self.Cs[0]),np.log10(1.0),len(self.Cs))[::-1]
                self.Fs = np.logspace(np.log10(self.Bs[0]/self.Cs[-1]),np.log10(2.0),len(self.Cs))




        self.nchan = nchan

        self.scattering_mod_f = None
        self.verbose = verbose
        self.levels = levels
        self.colors = colors
        self.lws = lws

    def template_fitting_error(self,S,Weff=100.0,Nphi=2048): #Weff in microseconds
        Nphi = 1 
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


    def build_template_fitting_cov_matrix(self,nus,T=1800.0,nuref=1.0,Tconst=20.0):
        '''
        
        '''
        
        Weffs = self.psrnoise.Weffs
        if type(Weffs) != np.ndarray:
            Weffs = np.zeros_like(nus)+Weffs
        B = self.get_bandwidths(nus)
       

        Tsys = Tconst + 20 * np.power(nus/0.408,-1*self.galnoise.beta)

        tau = 0.0
        if self.psrnoise.DM != 0.0 and self.psrnoise.D != 0.0 and self.galnoise.T_e != 0.0 and self.galnoise.fillingfactor != 0:
            tau = 1.417e-6 * (self.galnoise.fillingfactor/0.2)**-1 * self.psrnoise.DM**2 * self.psrnoise.D**-1 * np.power(self.galnoise.T_e/100,-1.35)

        numer =  (self.psrnoise.I_0 * 1e-3) * np.power(nus/nuref,-1*self.psrnoise.alpha)*np.sqrt(B*1e9*T) * np.exp(-1*tau*np.power(nus/nuref,-2.1)) 

        #denom = (2760.0 / self.psrnoise.A_e) * Tsys        
        denom = Tsys / self.telnoise.gain
        S = numer/denom
        sigmas = self.template_fitting_error(S,Weffs,1)

        if self.psrnoise.taud > 0.0:
            tauds = DISS.scale_tau_d(self.psrnoise.taud,nuref,nus)
            retval = self.scattering_modifications(tauds,Weffs)
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
        
    def build_jitter_cov_matrix(self):
        sigma_Js = self.psrnoise.sigma_Js
        if type(sigma_Js) != np.ndarray:
            sigma_Js = np.zeros(self.nchan)+sigma_Js

        retval = np.matrix(np.zeros((len(sigma_Js),len(sigma_Js))))
        if sigma_Js is not None:
            for i in range(len(sigma_Js)):
                for j in range(len(sigma_Js)):
                    retval[i,j] = sigma_Js[i] * sigma_Js[j]
        return retval

        
    def scattering_modifications(self,tauds,Weffs,filename="ampratios.npz",directory=None):
        if len(glob.glob(filename))!=1:
            if directory is None:
                directory = __file__.split("/")[0] + "/"
        else:
            directory = ""
        if type(Weffs) != np.ndarray:
            Weffs = np.zeros_like(nus)+Weffs

        if self.scattering_mod_f is None:
            data = np.load(directory+"ampratios.npz")
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
        

    def get_channels(self,nus,nuref=1.0,C1=1.16,T=1800.0,etat=0.2,etanu=0.2):

        #Bs = np.logspace(MIN,2*MAX,(2*MAX-MIN)*nsteps+1)

        numin = self.Bs[0]
        numax = self.Bs[-1]

        nus = self.Bs

        nsteps = 50
        numin = 0.01
        numax = 10.0
        MIN = np.log10(numin)
        MAX = np.log10(2*numax)
        #nus = np.logspace(MIN,2*MAX,(2*MAX-MIN)*nsteps+1) #2*MAX or np.log10(2*numax)?
        nus = np.logspace(MIN,MAX,(MAX-MIN)*nsteps+1) #2*MAX or np.log10(2*numax)?


        B = self.get_bandwidths(nus)
        dtd = DISS.scale_dt_d(self.psrnoise.dtd,nuref,nus)
        dnud = DISS.scale_dnu_d(self.psrnoise.dnud,nuref,nus)
        taud = DISS.scale_tau_d(self.psrnoise.taud,nuref,nus)

        niss = (1 + etanu* B/dnud) * (1 + etat* T/dtd) 

        for i,nu in enumerate(nus):
            print nu,niss[i]


        raise SystemExit
        pass
        
    def scintillation_noise(self,nus,nuref=1.0,C1=1.16,T=1800.0,etat=0.2,etanu=0.2):
        '''
        dtd0 in seconds
        dnud0 in GHz
        Uses an internal nsteps
        '''

        #self.get_channels(nus)

        numin = nus[0]
        numax = nus[-1]

        B = self.get_bandwidths(nus)
        dtd = DISS.scale_dt_d(self.psrnoise.dtd,nuref,nus)
        dnud = DISS.scale_dnu_d(self.psrnoise.dnud,nuref,nus)
        taud = DISS.scale_tau_d(self.psrnoise.taud,nuref,nus)

        niss = (1 + etanu* B/dnud) * (1 + etat* T/dtd) 

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
    def DM_misestimation(self,nus,errs,covmat=False):
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

        # Frequency-Dependent DM
        DM_nu_var = evalDMnuError(self.psrnoise.dnud,np.max(nus),np.min(nus))**2 / 25.0

        # PBF errors (scattering), included already in cov matrix?
        # Scattering error, assume this is proportional to nu^-4.4? or 4?
        chromatic_components = self.psrnoise.tauvar * np.power(nus,-4.4)
        scattering_var = np.dot(np.dot(np.dot(P,XT),VI),chromatic_components)[0,0]**2

        retval = np.sqrt(template_fitting_var + DM_nu_var + scattering_var)

        return retval





        
    def build_polarization_cov_matrix(self):
        W50s = self.psrnoise.W50s
        if type(W50s) != np.ndarray:
            W50s = np.zeros(self.nchan)+W50s
        if type(self.telnoise.epsilon) != np.ndarray:
            epsilon = np.zeros(self.nchan)+self.telnoise.epsilon
        if type(self.telnoise.pi_V) != np.ndarray:
            pi_V = np.zeros(self.nchan)+self.telnoise.pi_V
        if type(self.telnoise.eta) != np.ndarray:
            eta = np.zeros(self.nchan)+self.telnoise.eta
        if type(self.telnoise.pi_L) != np.ndarray:
            pi_L = np.zeros(self.nchan)+self.telnoise.pi_L



        sigmas = epsilon*pi_V*(W50s/100.0) #W50s in microseconds #do more?
        sigmasprime = 2 * np.sqrt(eta) * pi_L #Actually use this
        return np.matrix(np.diag(sigmas**2))



    def calc_single(self,nus):
        cov = self.build_template_fitting_cov_matrix(nus)

        jittercov = self.build_jitter_cov_matrix()
        disscov = self.scintillation_noise(nus) 
        cov = cov +jittercov + disscov
        sigma2 = epoch_averaged_error(cov,var=True)

        sigmatel2 = epoch_averaged_error(self.build_polarization_cov_matrix())

        sigma = np.sqrt(sigma2 + self.DM_misestimation(nus,cov,covmat=True)**2 + sigmatel2) #need to include PBF errors?

        if self.psrnoise.P is not None and sigma>self.psrnoise.P:
            return self.psrnoise.P

        return sigma


    def calc(self):
        self.sigmas = np.zeros((len(self.Cs),len(self.Bs)))
        if self.frac_bw == False:
            for ic,C in enumerate(self.Cs):
                if self.verbose:
                    print("Computing center freq %0.3f GHz (%i/%i)"%(C,ic,len(self.Cs)))
                for ib,B in enumerate(self.Bs):
                    if B > 1.9*C:
                        self.sigmas[ic,ib] = np.nan
                    else:
                        nulow = C - B/2.0
                        nuhigh = C + B/2.0

                        if self.log == False:
                            nus = np.linspace(nulow,nuhigh,self.nchan+1)[:-1] #more uniform sampling?
                        else:
                            nus = np.logspace(np.log10(nulow),np.log10(nuhigh),self.nchan+1)[:-1] #more uniform sampling?   

                        self.sigmas[ic,ib] = self.calc_single(nus)
        else:
            for ic,C in enumerate(self.Cs):
                print(ic,len(self.Cs),C)
                for indf,F in enumerate(self.Fs):
                    B = C*F
                    if B > 1.9*C or B <= 0:
                        self.sigmas[ic,indf] = np.nan
                    else:
                        nulow = C - B/2.0
                        nuhigh = C + B/2.0


                        if self.log == False:
                            nus = np.linspace(nulow,nuhigh,self.nchan+1)[:-1] #more uniform sampling?
                        else:
                            nus = np.logspace(np.log10(nulow),np.log10(nuhigh),self.nchan+1)[:-1] #more uniform sampling?   

                        self.sigmas[ic,indf] = self.calc_single(nus)



    def plot(self,filename="triplot.png",doshow=True,figsize=(8,6),save=True,minimum=None,points=None,colorbararrow=None):
        fig = figure(figsize=figsize)
        ax = fig.add_subplot(111)
        if self.frac_bw == False:
            data = np.transpose(np.log10(self.sigmas))
            if self.log == False:
                im = uimshow(data,extent=[self.Cs[0],self.Cs[-1],self.Bs[0],self.Bs[-1]],cmap=cm.inferno_r,ax=ax)

                ax.set_xlabel(r"$\mathrm{Center~Frequency~\nu_0~(GHz)}$")
                ax.set_ylabel(r"$\mathrm{Bandwidth}~B~\mathrm{(GHz)}$")
            else:

                im = uimshow(data,extent=np.log10(np.array([self.Cs[0],self.Cs[-1],self.Bs[0],self.Bs[-1]])),cmap=cm.inferno_r,ax=ax)
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
                ax.xaxis.set_major_formatter(noformatter)
                ax.yaxis.set_major_formatter(noformatter)

                ax.text(0.05,0.9,"PSR~%s"%self.psrnoise.name.replace("-","$-$"),fontsize=18,transform=ax.transAxes,bbox=dict(boxstyle="square",fc="white"))

            if minimum is not None:
                data = np.log10(self.sigmas)
                flatdata = data.flatten()
                inds = np.where(np.logical_not(np.isnan(flatdata)))[0]
                MIN = np.min(flatdata[inds])
                INDC,INDB = np.where(data==MIN)
                INDC,INDB = INDC[0],INDB[0]
                MINB = self.Bs[INDB]
                MINC = self.Cs[INDC]
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
                        sigma = np.log10(self.calc_single(nus))
                    else:
                        ax.plot(x,y,fmt,zorder=50,ms=8)
                        nus = np.linspace(nulow,nuhigh,self.nchan+1)[:-1] #more uniform sampling?
                        sigma = np.log10(self.calc_single(nus))
                    with open("minima.txt",'a') as FILE:
                        FILE.write("%s point %f %f %f\n"%(self.psrnoise.name,x,y,sigma))




            if colorbararrow is not None:
                data = np.log10(self.sigmas)
                flatdata = data.flatten()
                inds = np.where(np.logical_not(np.isnan(flatdata)))[0]
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

                im = uimshow(data,extent=np.log10(np.array([self.Cs[0],self.Cs[-1],self.Fs[goodinds][0],self.Fs[goodinds][-1]])),cmap=cm.inferno_r,ax=ax)
                cax = ax.contour(data,extent=np.log10(np.array([self.Cs[0],self.Cs[-1],self.Fs[goodinds][0],self.Fs[goodinds][-1]])),colors=COLORS,levels=LEVELS,linewidths=LWS,origin='lower')
                

                ax.set_xlabel(r"$\mathrm{Center~Frequency~\nu_0~(GHz)}$")
                #ax.set_ylabel(r"$r~\mathrm{(\nu_{max}/\nu_{min})}$")
                ax.set_ylabel(r"$\mathrm{Fractional~Bandwidth~(B/\nu_0)}$")
                ax.yaxis.set_major_locator(FixedLocator(np.log10(np.arange(0.25,1.75,0.25))))
                ax.xaxis.set_major_formatter(noformatter)
                ax.yaxis.set_major_formatter(noformatter)
            
            
        cbar = fig.colorbar(im)#,format=formatter)
        cbar.set_label("$\mathrm{TOA~Uncertainty~\sigma_{TOA}~(\mu s)}$")

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
        if self.frac_bw == False:
            np.savez(filename,Cs=self.Cs,Bs=self.Bs,sigmas=self.sigmas)
        else:
            np.savez(filename,Cs=self.Cs,Fs=self.Fs,sigmas=self.sigmas)





