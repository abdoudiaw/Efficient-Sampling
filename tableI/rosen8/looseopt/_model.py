#!/usr/bin/env python
#
# Author: Mike McKerns (mmckerns @uqfoundation)
# Copyright (c) 2021 The Uncertainty Quantification Foundation.
# License: 3-clause BSD.  The full license text is available at:
#  - https://github.com/uqfoundation/mystic/blob/master/LICENSE

mname = 'eval'  # 'model' name
ename = 'stop'  # termination name

shape = (8,0) #NOTE: z is single-valued

etol = 1 # None # int rounding precision for termination point cache
npts = 16 #125 #2 #500 # number of solvers used in a searcher
direct = True # optimizer-directed sampling or not
lower = 0 # lower bound on input parameters
upper = 10 # upper bound on input parameters
bounds = [(lower,upper)] * shape[0] # bounds (lower is zero) on input parameters

# define hypercube for plotting
hcube = '{0}:{1}:{2}, {0}:{1}:{2}'.format(lower, upper, upper/50.)
hcube += ', 1.0, 1.0, 1.0, 1.0, 1.0, 1.0'
scale = None #1 # plot depth scaling

# controls
warms = 1000 # this is size of 'warm'
fails = 10 # this is number of refit tries before 'fail' to sampling
maxtol = 1e-4 # max tol for valid
avetol = 1e-5 # ave tol for valid
reptol = 2e-4 # stop when rep converges so rep iters <= reptol
nrep = 3 # this is hist.values()[-rep:] <= reptol
nlast = 3 # this is any(size[-nlast:])
maxloops = 30 # max number of loops

# objective
from mystic.models import dejong
objective = dejong.Rosenbrock(ndim=shape[0]).function

# select solver and searcher algorithms
from mystic.solvers import NelderMeadSimplexSolver as solver
from mystic.samplers import *
searcher = SparsitySampler
#searcher = BuckshotSampler
