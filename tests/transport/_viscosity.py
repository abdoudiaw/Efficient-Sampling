import numpy as np

def viscosity(x):
    """
    Computes dimensionless viscosity -
    Daligault PRE (2010)
    Input:
      - x -- array of model parameters: material temperature and density
          -- density is normalized to 1e22 1/cm^3
          -- temperature is normalized to  300 eV
      
    Output:
      - reduced viscosity -- viscosity in units of rho wp^2a^2
      """
    
    a=[0.794811,0.0425698, 0.00205782,7.03658e-5]
    b=[0.862151, 0.0429942, -0.000270798, 3.25441e-6, -1.15019e-8]

    Gamma=1.93*x[1]**(1./3)/x[0]      # calculates the coupling parameter

    factor      = a[0]/(Gamma**2.5*np.log(1.+b[0]/Gamma**1.5))
    numerator   = 1.+a[1]*Gamma+a[2]*Gamma**2+a[3]*Gamma**3
    denominator = 1.+b[1]*Gamma+b[2]*Gamma**2+b[3]*Gamma**3+b[4]*Gamma**4
    
    return factor*numerator/denominator
