import numpy as np

def gstr(x):
    r=x[0] #*5
    Gamma=x[1]*6+0.001 #*20
    
    #Weakly coefficients coupled
    phi=Gamma*(1.+0.167*Gamma**(4./3.))
    eta=np.sqrt(3.*Gamma)*(1.0-0.1495*np.sqrt(Gamma))
    Xi=1.826e-2*Gamma**(4./3.)
    rho=min(2.50,1.378+0.284/Gamma)
    tau=min(1.25,0.505+0.289/Gamma**(2./3))
    
    #Strongly coefficients coupled
    psi=1.634+7.934e-3*np.sqrt(Gamma)+(1.608/Gamma)**2
    sigma=1.+1.093e-2*(np.log(Gamma))**3
    mu= 0.246+3.145*Gamma**(3./4.)
    nu=2.084+1.706/np.log(Gamma)
    alpha = 6.908+(0.860/Gamma)**(1./3.)
    beta= 0.231-1.785*np.exp(-Gamma/60.2)
    gam = 0.140+0.215*np.exp(-Gamma/14.6)
    delta= 3.733+2.774*Gamma**(1./3.)
    eps = 0.993 +(33./Gamma)**(2./3.)
    x=r/psi-1.
    
    Gammax=(delta-eps)*np.exp(-np.sqrt(x/gam))+eps
    
    #Strongly coupled
    k1=sigma*np.exp(-mu*(-x)**nu)
    k2=1.+(sigma-1.)*np.cos(alpha*x+beta*np.sqrt(x))/np.cosh(x*Gammax)
    gstrong=np.where(x <= 0., k1, k2)
    
    #Weakly coupled
    gweak=np.exp(-phi/r*np.exp(-eta*r))+Xi*np.exp(-(r-rho)**2/tau**2)
    if Gamma <=5.:
        return gweak
    else:
        return gstrong
