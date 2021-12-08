#!/usr/bin/env python
#
# Author: Mike McKerns (mmckerns @uqfoundation)
# Copyright (c) 2021 The Uncertainty Quantification Foundation.
# License: 3-clause BSD.  The full license text is available at:
#  - https://github.com/uqfoundation/mystic/blob/master/LICENSE

mname = 'eval'  # 'model' name
ename = 'stop'  # termination name

shape = (3,0) #NOTE: z is single-valued

etol = 1 # None # int rounding precision for termination point cache
npts = 4 #125 #2 #500 # number of solvers used in a searcher
direct = True # optimizer-directed sampling or not
lower = -3 # lower bound on input parameters
upper = 3 # upper bound on input parameters
bounds = [(lower,upper)] * shape[0] # bounds (lower is zero) on input parameters

# define hypercube for plotting
hcube = '{0}:{1}:{2}, {0}:{1}:{2}'.format(lower, upper, upper/50.)
hcube += ', 1.0'
scale = None #1 # plot depth scaling

# controls
warms = 100 # this is size of 'warm'
fails = 50 # this is number of refit tries before 'fail' to sampling
maxtol = 1e-6 # max tol for valid
sumtol = 1e-3 # ave tol for valid
reptol = 1e-3 #FIXME: 2e-5 # stop when rep converges so rep iters <= reptol
nrep = 1 # this is hist.values()[-rep:] <= reptol
nlast = 1 # this is any(size[-nlast:])
maxloops = 1 #30 # max number of loops

# objective
from mystic.models import dejong
objective = dejong.Rosenbrock(ndim=shape[0]).function

# select solver and searcher algorithms
from mystic.solvers import PowellDirectionalSolver as solver
from mystic.samplers import *
#searcher = LatticeSampler
searcher = BuckshotSampler
