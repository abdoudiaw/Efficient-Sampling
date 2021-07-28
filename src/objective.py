#!/usr/bin/env python
#
# Author: Mike McKerns (mmckerns @lanl.gov and @uqfoundation)
# Copyright (c) 2019-2021 Los Alamos National Laboratory.
# License: 3-clause BSD.  The full license text is available at:
#  - file://LICENSE
"""
toy surrogate models for LAMMPS
"""

import numpy as np

def objective0(x):
  "N dimensional model"
  return LAMMPS1D(*convert(x))[0]


def objective1(x, axis=None):
  "Nx1 dimensional model"
  if axis is None: axis = slice(None)
  return LAMMPS1D(*convert(x))[axis]


def objective2(x, axis=None):
  "Nx2 dimensional model"
  if axis is None: axis = slice(None)
  return LAMMPS2D(*convert(x))[axis]


def objective3(x, axis=None):
  "Nx3 dimensional model"
  if axis is None: axis = slice(None)
  return LAMMPS3D(*convert(x))[axis]


def convert(x):
  "convert parameter array to temperature, densities, Zs"
  x = np.asarray(x)
  lx = len(x[1:])
  n = 1 + int(lx/2)
  return (x[0], x[1:n], x[n:-1 if lx%2 else None])

#NOTE: temperature is between (0,1). Otherwise (0,inf),where inf is ~10

def LAMMPS1D(temperature, densities, Z):
  "a (1,N,N)x1 dummy function with the same interface as call_LAMMPS"
  y = np.sum([densities[i]*Z[i] for i in range(len(Z))])/len(Z)
  return y**2 + temperature**2,


def LAMMPS2D(temperature, densities, Z):
  "a (1,N,N)x2 dummy function with the same interface as call_LAMMPS"
  y = np.sum([densities[i]*Z[i] for i in range(len(Z))])/len(Z)
  x = np.sum([max(densities)*Z[i] for i in range(len(Z))])/len(Z)
  w = np.prod([list(densities[i]*Z[i] for i in range(len(Z)))])/len(Z)
  return y**2 + temperature**2, 2*x**2 + (w-temperature)**2


def LAMMPS3D(temperature, densities, Z):
  "a (1,N,N)x3 dummy function with the same interface as call_LAMMPS"
  y = np.sum([densities[i]*Z[i] for i in range(len(Z))])/len(Z)
  x = np.sum([max(densities)*Z[i] for i in range(len(Z))])/len(Z)
  w = np.prod([list(densities[i]*Z[i] for i in range(len(Z)))])/len(Z)
  return y**2 + temperature**2, 2*x**2 - temperature**2, (w-temperature)**2


# Takes our thermodynamic input parameters and returns two reduced quantities
# Gamma and Kappa. Gamma is the ratio between kinetic and potential energy
# while Kappa estimates strength of screening around an ion due to others ions.

def kernel(x):
  "kernel to transform to a 2-dimensional space"
  Temperature, density, Z = convert(x)
  density = sum(density)

  Debye_length = 7.43e2 * np.sqrt(Temperature/density)
  aws = (3./(4. * np.pi * density))**(1./3.)

  Kappa = aws/Debye_length
  Gamma = 1.44e-07/(Temperature*aws)

  return Gamma, Kappa

