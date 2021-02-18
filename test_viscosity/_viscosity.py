import numpy as np

def viscosity(state_parameters):

    """compute the viscosity of dense plasma
    Input:
      - state_parameters -- array of model parameters: material temperature and density
          --density is normalized to 1e22 1/cm^3
          --temperature is normalized to  300 eV
      
    Output:
      - reduced viscosity -- viscosity in units of rho wp^2a^2
      """

    a=np.array([0.794811,0.0425698, 0.00205782,7.03658e-5])
    b=[0.862151, 0.0429942, -0.000270798, 3.25441e-6, -1.15019e-8]

    p=1.93*state_parameters[1]**(1./3)/state_parameters[0]      # calculates the coupling parameter

    T1=a[0]/(p**2.5*np.log(1.+b[0]/p**1.5))
    T2=(1.+a[1]*p+a[2]*p**2+a[3]*p**3)/(1+b[1]*p+b[2]*p**2+b[3]*p**3+b[4]*p**4)

    return T1*T2
