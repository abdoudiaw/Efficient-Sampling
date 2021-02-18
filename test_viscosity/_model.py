
shape = (2,0) #NOTE: z is single-valued

mname = 'demo'  # dynamic 'model' name
fname = 'dirty' # static 'data' name
pname = 'demo'  # hyper_plot name

npts = 4   # number of solvers used in a searcher
upper = 1  # upper bound on input parameters
#bounds = [(0.,1),(0.5,0.5)] # * shape[0] # bounds (lower is zero) on input parameters
bounds = [(0.01,1),(0.1,1)] # * shape[0] # bounds (lower is zero) on input parameters

#(counter_yp=x[0],counter_np=x[1])

# objective
#from skyrme_bag_hybrid import pressure as model
from _viscosity import viscosity as model

objective = lambda x,**kwds: model(x)

# select solver and searcher algorithms
#from mystic.solvers import PowellDirectionalSolver as solver

from mystic.solvers import NelderMeadSimplexSolver as solver
from sampler import *
#searcher = LatticeSampler
searcher = BuckshotSampler

