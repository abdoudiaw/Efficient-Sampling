import numpy as np
import subprocess as sp
import csv
import os
from scipy import interpolate
import mystic.cache as mc
from _model import *

aws=2.87941216927844e-06
        
def OCP(x):

    """OCP RDF from the MD code LAMMPS
    
    Inputs:
      x: [r,Gamma] an array of the radial coordinate and plasma coupling
    Output:
        z: g(r) - radial distribution function
    """
    r_,Gamma = x
    with open('Gamma.csv', 'w') as testfile:
        csv_writer = csv.writer(testfile,delimiter=' ')
        csv_writer.writerow([Gamma])
    # Load MD script here
    lammps_script = open('in.ocp')
    #  Run the MD
    args=['mpirun','-np', '4','lmp_mpi']
    sp.Popen(args, stdin=lammps_script).wait()
    
   # Load the MD results, interpolate and return result for a given r
    data=np.loadtxt('rdf.csv', skiprows=4, unpack=True)
    f = interpolate.interp1d(data[1]/aws, data[2])

   # Save MD data to Klepto DB
    db = mc.archive.read(mname)
    gamma_values = np.full((len(data[1])), Gamma, dtype=float)
    x=[i for i in zip(data[1]/aws,gamma_values)]
    z = zip(x,data[2])
    print('len z', z)
    mc.archive.write(db, z)
    
    return f(r_)


