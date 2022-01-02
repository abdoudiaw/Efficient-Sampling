import numpy as np
from scipy.constants import pi

from numba import jit
import random


     
# Basic physics parameters
hc  = 1.973269631e2                            # hbar x c [MeV fm]
hc3 = hc*hc*hc                                 # (hbar x c)^3, often used
c   = 2.99792458e10                            # speed of light
me  = 0.510999                                 # electron mass [MeV]

# Nucleon parameters
mn  = 939.565560                               # neutron mass [MeV]
mp  = 938.27201                                # proton mass [Mev]
mm  = (mn+mp)/2.0                              # nucleon mass [MeV]

# Quark parameters
bc  = 170.0                                    # MIT Bag constant [MeV]
bag = bc*bc*bc*bc/(hc*hc*hc)                   # MIT Bag constant [MeV/fm^3]
ms  = 150.0                                    # Strange quark mass [MeV]
mu  = 0.0                                      # Up quark mass (massless)
md  = 0.0                                      # Down quark mass (massless)

# Skyrme EoS Parameter
t0 = -2488.91
t1 =  486.82
t2 = -546.39
t3 =  13777.0
x0 =  0.834
x1 = -0.344
x2 = -1.0
x3 =  1.354
W0 = 123.0
sigma = 1.0/6.0

# Function F
@jit(nopython=True)
def F(pow, yp):
    return np.power(2.0,pow-1.0)*(np.power(yp,pow)+np.power(1.0-yp,pow))

# Electrons
# Input: Proton fraction, baryon number density
# Ouput: Energy density, pressure, electron chemical potential
@jit(nopython=True)
def electron(yp,nb):
    n_e   = yp*nb                                                   # Electron density
    kf    = np.power(n_e*(3.0*pi*pi*hc3),1.0/3.0)                   # Electron fermi momentum
    mue   = np.sqrt(kf*kf+me*me)                                    # Electron chemical potential
    j     = kf/me
    x     = np.log(j+np.sqrt(1.0+j*j))
    e01   = np.power(me,4.0)/(pi*pi*hc3)
    eng_e = (e01/8.0)*((np.sqrt(1.0+j*j))*j*(1.0+2.0*j*j)-x)        # Electron energy density
    p_e   = (e01/24.0)*((np.sqrt(1.0+j*j))*j*(2.0*j*j-3.0) \
          + 3.0*np.log(j+np.sqrt(1.0+j*j)))                         # Electron pressure
    return eng_e, p_e, mue


# Quarks
# Input: Bag constant, baryon chemical potential, proton fraction
# Output: Energy density, pressure, quark number density
@jit(nopython=True)
def quark(yp,bag,mub,muc):
    muq = mub/3.0

    # up quark number density and pressure
    muu = muq + 2.0*muc/3.0
    nu  = muu*muu*muu/(pi*pi)
    p_u = muu*muu*muu*muu/(4.0*pi*pi)

    # down quark number density and pressure
    mud = muq - muc/3.0
    nd  = mud*mud*mud/(pi*pi)
    p_d = mud*mud*mud*mud/(4.0*pi*pi)

    # strange quark number density and pressure
    mus = muq - muc/3.0
    kf  = 0.0
    if (np.abs(mus) > ms):
        kf = np.sqrt(mus*mus - ms*ms)
    j   = kf/ms
    x   = np.log(j+np.sqrt(1+j*j))
    e01 = ms*ms*ms*ms/(pi*pi)
    ns  = kf*kf*kf/(pi*pi)
    p_s = (3.0*e01/24.0)*(np.sqrt(1.0+j*j)*j*(-3.0+2.0*j*j)+3.0*x)

    # sum up, down, and strange quark contributions
    p_q   = ((p_s+p_u+p_d)/hc3)-bag
    eng_q = 3.0*p_q + 4.0*bag
    nb    = (ns+nd+nu)/(3.0*hc3)
    nc    = (2.0*nu/3.0-nd/3.0-ns/3.0)/hc3

    # add electron contribution
    eng_e, p_e, mue = electron(yp,nb)
    pressure = p_q + p_e
    energy = eng_q + eng_e

    # calculate proton and neutron chemical potentials
    mup = 2.0*muu + mud
    mun = 2.0*mud + muu

    # effective chemical potential for phase transition
    mueff_q = (mue + mup)*(nc/nb) + mun*(1.0 - (nc/nb))
    return energy, pressure, nb, nc, mueff_q


# Root-searching routine for quark phase
# Input: bag constant, baryochemical potential, proton fraction
# Output: effective quark potential, charge chemical potential

def muc_quark(bag,mub,yp):
    mucu = -1.5*(mub/3.0) + 1.0
    muco =  3.0*(mub/3.0) - 1.0
      
    muc = mucu
    eng_q, p_q, nb, nc, mueff_q = quark(yp,bag,mub,muc)
    yu = (nc/nb)-yp

    muc = muco
    eng_q, p_q, nb, nc, mueff_q = quark(yp,bag,mub,muc)
    yo = (nc/nb)-yp

    cont = yu*yo
    if (cont > 0.0):
        print("error in root-finding routine")
        print(yu,yo,cond,yp,mub)
        print("Choose new intervall for muc")
            
    eps = 1.e-10
    acc = (muco-mucu)/2.0
            
    while (acc>eps):
        muc_new = (muco+mucu)/2.0
         
        muc = mucu
        eng_q, p_q, nb, nc, mueff_q = quark(yp,bag,mub,muc)
        yu = (nc/nb)-yp
                  
        muc = muc_new
        eng_q, p_q, nb, nc, mueff_q = quark(yp,bag,mub,muc)
        ynew = (nc/nb)-yp
          
        if (yu*ynew<0):
            muco = muc_new
        if (yu*ynew>0):
            mucu = muc_new
        acc = acc/2.0

    muc = (muco+mucu)/2.0
    ng_q, p_q, nb, nc, mueff_q = quark(yp,bag,mub,muc)
    return mueff_q, muc, nb


# Root-searching routine for quark phase
# Input: bag constant, effective potential, proton fraction
# Ouput: baryochemical potential, charge chemical potential
def chemp_quark(bag,mueff,yp):
    xu = 50.0
    xo = 2000.0
       
    mub = 3.0*xu
    mueff_q, muc_q, nb = muc_quark(bag,mub,yp)
    mueff_u = mueff_q - mueff

    mub = 3.0*xo
    mueff_q, muc_q, nb = muc_quark(bag,mub,yp)
    mueff_o = mueff_q - mueff
    
    cond = mueff_u*mueff_o
    if (cond>0):
        print ("Error in chemp_quark routine")
        print (mueff_u,mueff_o,mueff)
        print("Choose new intervall for mub")
    
    eps = 1.e-5
    acc = (xo-xu)/2.0
    
    while (acc>eps):
      xnew = (xo+xu)/2.0
      
      mub = 3.0*xu
      mueff_q, muc_q, nb = muc_quark(bag,mub,yp)
      mueff_u = mueff_q - mueff
                
      mub = 3.0*xnew
      mueff_q, muc_q, nb = muc_quark(bag,mub,yp)
      mueff_new = mueff_q - mueff
      
      if(mueff_u * mueff_new<0):
          xo = xnew
      
      if(mueff_u * mueff_new>0):
          xu = xnew
      
      acc = acc/2.0

    mub = 3.0*(xo+xu)/2.0
    mueff_q, muc_q, nb = muc_quark(bag,mub,yp)
    return mub, muc_q


# Root-searching routine for quark phase
# Input: bag constant, baryon number density, proton fraction
# Ouput: baryochemical potential, charge chemical potential
def nb_quark(bag,nb,yp):
    xu = 50.0
    xo = 2000.0
       
    mub = 3.0*xu
    mueff_q, muc_q, nb_q = muc_quark(bag,mub,yp)
    nb_u = nb_q - nb

    mub = 3.0*xo
    mueff_q, muc_q, nb_q = muc_quark(bag,mub,yp)
    nb_o = nb_q - nb
    
    cond = nb_u*nb_o
    if (cond>0):
        print ("Error in nb_quark routine")
        print (nb_u,nb_o,nb)
        print("Choose new intervall for mub")
    
    eps = 1.e-5
    acc = (xo-xu)/2.0
    
    while (acc>eps):
      xnew = (xo+xu)/2.0
      
      mub = 3.0*xu
      mueff_q, muc_q, nb_q = muc_quark(bag,mub,yp)
      nb_u = nb_q - nb
                
      mub = 3.0*xnew
      mueff_q, muc_q, nb_q = muc_quark(bag,mub,yp)
      nb_new = nb_q - nb
      
      if(nb_u * nb_new<0):
          xo = xnew
      
      if(nb_u * nb_new>0):
          xu = xnew
      
      acc = acc/2.0

    mub = 3.0*(xo+xu)/2.0
    mueff_q, muc_q, nb_q = muc_quark(bag,mub,yp)
    return mub, muc_q, nb_q


# Hadrons (neutron, protons, ...)
# Input: Proton fraction, baryon number density
# Output: Energy density, pressure
@jit(nopython=True)
def hadronic(yp,nb):
    Func1 = F(5.0/3.0,yp)
    Func2 = F(2.0,yp)
    Func3 = F(8.0/3.0,yp)
    A1 = (3.0*hc*hc/(10.0*mm))*np.power(1.5*pi*pi,2.0/3.0)*Func1
    A2 = (t0/8.0)*(2.0*(x0+2.0)-(2.0*x0+1.0)*Func2)
    A3 = (t3/48.0)*(2.0*(x3+2.0)-(2.0*x3+1.0)*Func2)
    E  = t1*(x1+2.0)+t2*(x2+2.0)
    G  = 0.5*(t2*(2.0*x2+1.0)-t1*(2.0*x1+1.0))
    A4 = (3.0/40.0)*np.power(1.5*pi*pi,2.0/3.0)*(E*Func1+G*Func3)

    # calculate nucleonic pressure
    p_nucl = A1*np.power(nb,5.0/3.0)*(2.0/3.0) + \
             A2*nb*nb + A3*np.power(nb,sigma+2.0)*(sigma+1.0) + \
             A4*np.power(nb,8.0/3.0)*(5.0/3.0)
    # calculate nucleonic energy density
    eng_nucl = A1*np.power(nb,5.0/3.0) + A2*nb*nb +\
               A3*np.power(nb,sigma+2.0) + A4*np.power(nb,8.0/3.0) +\
               yp*mp*nb+(1.0-yp)*mn*nb
    # calculate energy per baryon
    EA = (A1*np.power(nb,5.0/3.0) + A2*nb*nb + A3*np.power(nb,sigma+2.0) +\
          A4*np.power(nb,8.0/3.0))/nb

    UN = 2.0*A2*nb + (sigma+2.0)*A3*np.power(nb,sigma+1.0) +\
        (8.0/3.0)*A4*np.power(nb,5.0/3.0)

    # calcualte symmetry energy
    Theta = 3.0*t1*x1-t2*(5.0+4.0*x2)
    S = (hc*hc/(6.0*mm))*np.power(1.5*pi*pi*nb,2.0/3.0) - (t0/8.0)*(2.0*x0+1.0)*nb -\
        (1.0/24.0)*np.power(1.5*pi*pi*nb,2.0/3.0)*Theta*nb -\
        (t3/48.0)*(2.0*x3+1.0)*np.power(nb,sigma+1)

    # approximate for mun - mup = 4xSx(1-yp)
    mun_mup = 4.0*S*(1.0-2.0*yp)
    mub = ((eng_nucl + p_nucl)/nb) + (mun_mup)*yp
    mup = mub - 4.0*S*(1.0-2.0*yp)

    # add electron contribution
    eng_e, p_e, mue = electron(yp,nb)
    pressure = p_nucl + p_e
    energy = eng_nucl + eng_e
    mueff = (mue + mup)*yp + mub*(1.0-yp)
    return energy, pressure, mub, mueff
    

# Phase transition determination
# Input: Proton fraction
# Output: Critical number densities and and pressure
def phase_transition(yp,nb_e):
    counter_nb = 0
    nb_crit = 2.0*nb_e
    while (nb_crit > nb_e) and (counter_nb < nb_num):
        nb = nb_i*(np.power(10.0,dnb*counter_nb))
        eng_h, p_h, mub, mueff = hadronic(yp,nb)

        mub_q, muc_q = chemp_quark(bag,mueff,yp)
        eng_q, p_q, n_q, nc, mueff_q = quark(yp,bag,mub_q,muc_q)

        if ((p_q > p_h) and (mub > 945)):
            p_crit_h = p_h
            p_crit_q = p_q
            p_crit = (p_q + p_h)/2.0
            nb_crit = nb
            nb_crit_q = n_q
        else:
            p_crit = 0.0
            nb_crit = 2.0*nb_e
            nb_crit_q = 2.0*nb_e

        counter_nb = counter_nb + 1
    
    return p_crit_h,p_crit_q,nb_crit,nb_crit_q


# ------------
# Main routine
# This part loops over the proton fraction and baryon number
# densities and determine if matter is in the nucleonic phase
# (i.e. neutrons and protons), mixed phase, or quark phase.
# Depending on the phase the code determines the pressure
#
# The nucleonic matter is modelled by the SLy4 Skyrme model.
# The quark phase is modelled by the MIT Bag model with a bag
# constant of B = 170 MeV.
# The mixed phase is modelled as a Maxwell phase transition
# where the equilibrium of nucleonic and quark phase happens at
# an effective chemical potential mu_n (1-yp) + (mu_p + mu_e)yp
# (see Hempel et al. PRD 80 (2009) for details).
# ------------

nb_num = 100                                        # Number of density steps
nb_i   = 0.04                                       # Initial number density [fm^-3]
nb_e   = 1.6                                        # Final number density [fm^-3]
dnb    = np.log10(nb_e/nb_i)/nb_num                 # Step size in number density

yp_num = 50                                         # Number of proton fraction steps
yp_i   = 0.0                                        # Initial proton fraction
yp_e   = 0.6                                        # Final proton fraction
dyp    = (yp_e - yp_i)/yp_num                       # Step size in proton fraction

print("# ")
print("# Phase transition determination between Sly4 Skyrme and MIT Bag models")
print("# Bag constant [MeV]:",bc)
print("# s-quark mass [MeV]:",ms)
print("# col1: yp; col2: nb [1/fm^3]; col3: pressure [MeV/fm^3]")
print("# ")


x=[]; y=[]
# Loop over proton fraction
#for counter_yp in range(yp_num):

def pressure(x):
    counter_yp=x[0]
    counter_np=x[1]


    yp = yp_i + dyp*counter_yp*yp_num
    # determine critical (i.e. phase transition) densities and pressures for given yp
    p_crit_h, p_crit_q, nb_crit_h, nb_crit_q = phase_transition(yp,nb_e)

    # loop over number density

    nb = nb_i*(np.power(10.0,dnb*counter_np*nb_num))

        # phase transition at equal effective chemical potentials
        # mueff = (mue + mup)*yp + mun*(1-yp)
    eng_h, p_h, mub, mueff = hadronic(yp,nb)
    mub_q,muc_q = chemp_quark(bag,mueff,yp)
    eng_q, p_q, n_q, n_c, mueff_q = quark(yp,bag,mub_q,muc_q)

        # nucleonic phase
    if (nb < nb_crit_h):
        pressure = p_h
        # quark phase
    elif (nb > nb_crit_q):
        mub_q, muc_q, nb_q = nb_quark(bag,nb,yp)
        eng_q, p_q, n_q, n_c, mueff_q = quark(yp,bag,mub_q,muc_q)
        pressure = p_q
        # mixed phase
    else:
        chi = (nb - nb_crit_h)/(nb_crit_q - nb_crit_h)
        pressure = (1.0 - chi)*p_crit_h + chi*p_crit_q
        
    return pressure
