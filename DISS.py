'''
n_ISS
From Cordes and Chernoff 1997:
nISS ~ (1+0.2 B/dnu_d)(1+0.2T/dt_d)
where B is the bandwidth, T is the integration time,
dnu_d is the characteristic bandwidth of DISS,
dt_d is the characteristic timescale
'''
def nISS(B,dnu_d,T,dt_d,eta=0.2):
    return (1+eta*B/dnu_d)*(1+eta*T/dt_d)
n_ISS = nISS

def sigma_DISS(tau_d,n_ISS):
    return tau_d/(n_ISS)**0.5
    

'''
Scale dnu_d and dt_d based on:
dnu_d propto nu^(22/5)
dt_d propto nu^(6/5) / transverse velocity
See Stinebring and Condon 1990 for scalings with beta (they call it alpha)

Be careful with float division now. This has been removed to allow for numpy arrays to be passed through.
'''
KOLMOGOROV_BETA = 11.0/3

def scale_dnu_d(dnu_d,nu_i,nu_f,beta=KOLMOGOROV_BETA):
    if beta < 4:
        exp = 2.0*beta/(beta-2) #(22.0/5)
    elif beta > 4:
        exp = 8.0/(6-beta)
    return dnu_d*(nu_f/nu_i)**exp
scale_dnud = scale_dnu_d

#Include parameters for changing velocity!
def scale_dt_d(dt_d,nu_i,nu_f,beta=KOLMOGOROV_BETA):
    if beta < 4:
        exp = 2.0/(beta-2) #(6.0/5)
    elif beta > 4:
        exp = float(beta-2)/(6-beta)
    return dt_d*(nu_f/nu_i)**exp
scale_dtd = scale_dt_d

def scale_tau_d(tau_d,nu_i,nu_f,beta=KOLMOGOROV_BETA):
    if beta < 4:
        exp = -2.0*beta/(beta-2) #(-22.0/5)
    elif beta > 4:
        exp = -8.0/(6-beta)
    return tau_d*(nu_f/nu_i)**exp
scale_taud = scale_tau_d

def scale_dt_r(tau_d,nu_i,nu_f,beta=KOLMOGOROV_BETA):
    if beta < 4:
        exp = beta/float(2-beta) #-2.2
    elif beta > 4:
        exp = 4.0/(beta-6)
    return tau_d*(nu_f/nu_i)**exp
scale_dtr = scale_dt_r
