#!/usr/bin/env python
#
# Author: Mike McKerns (mmckerns @lanl.gov and @uqfoundation)
# Copyright (c) 2019-2021 Los Alamos National Laboratory.
# License: 3-clause BSD.  The full license text is available at:
#  - file://LICENSE

#shape = (5,3)
#shape = (3,2)
#FIXME: shape = (5,0) #NOTE: z is single-valued
shape = (2,0) #FIXME: rastrigin

mname = 'demo'  # dynamic 'model' name
fname = 'dirty' # static 'data' name
pname = 'demo'  # hyper_plot name
ename = 'stop'  # termination name

etol = 1 # None # int rounding precision for termination point cache
npts = 4 # 125 #125 #2 #500 # number of solvers used in a searcher
upper = 10 #FIXME: 1000  # upper bound on input parameters
bounds = [(0.01,upper)] * shape[0] # bounds (lower is zero) on input parameters

# objective
if shape[-1]:
    exec('from objective import objective{0} as objective'.format(shape[-1]))
else: # z is single-valued
#    from mystic.models import rastrigin as objective
    from dca import gstr as objective
    #from perlin_nick import MultiscaleNDFunc
    #objective = MultiscaleNDFunc(1, 20, 100, 1, 2)

# select solver and searcher algorithms
from mystic.solvers import NelderMeadSimplexSolver as solver
from mystic.samplers import *
searcher = SparsitySampler
#searcher = BuckshotSampler
