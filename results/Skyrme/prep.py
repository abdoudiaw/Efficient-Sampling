#!/usr/bin/env python
#
# Author: Mike McKerns (mmckerns @uqfoundation)
# Copyright (c) 2019-2021 The Uncertainty Quantification Foundation.
# License: 3-clause BSD.  The full license text is available at:
#  - https://github.com/uqfoundation/mystic/blob/master/LICENSE

from _model import *

# apply cache/archive in 3D same as 1D
import mystic.cache as mc

# produce objective function that caches multi-valued output
model = mc.cached(archive=mname, multivalued=bool(shape[-1]))(objective)

# get the cache (it stores the full multi-valued tuple)
cache = model.__cache__()

# get "model inverse" (for maximization)
imodel = model.__inverse__

# produce memoization function to cache 'solved' points 
memo = mc.cached(archive=ename, tol=etol, multivalued=bool(shape[-1]))(lambda x, **kwds: kwds['out'])

# get the cache (for solved values)
extrema = memo.__cache__()

#XXX (apply logger?)

def sample(axis=None, invert=False):
    """search (for minima) until terminated
    """#FIXME: {solver:directed or None}, {terminated:all, evals:1000}
    if invert:
        _model = imodel
        l = -1
    else:
        _model = model
        l = 1
    if axis is None:
        model_ = lambda x: _model(x)
    else:
        model_ = lambda x: _model(x, axis)
    s = searcher(bounds, model_, npts=npts, solver=solver,rtol=0.1)
    #print('sampling...')
    if direct:
        s.sample_until(terminated=all) # npts=4 ==> ~800
    else:
        s.sample_until(evals=1) # npts=500 -> 1000
    #print('done sampling...')
    #NOTE: extract and save last points in the solver (nominally, the extrema)
    if etol is not None: #XXX: use None to turn off caching 'solved' values?
        slv = s._sampler._allSolvers
        for _s in slv:
            memo(_s.bestSolution, out=l*_s.bestEnergy)
    return s

def isample(axis=None):
    """search (for maxima) until terminated
    """#FIXME: {solver:directed or None}, {terminated:all, evals:1000}
    return sample(axis=axis, invert=True)

def _apply(f, arg):
    """call a function f with one argument arg"""
    return f(arg)

def search(axis, **kwds): #FIXME: axis=None, samplers=None
    """search for minima and maxima, until terminated

    Inputs:
      axis: int in [0,N], the axis of z to select
      map: (parallel) map function 
    """ #XXX: other kwds to pass to sample/isample
    _map = kwds.get('map', map)
    fs = (sample, isample) #FIXME: accept list of samplers (don't hardwire)
    return list(_map(_apply, fs, [axis]*len(fs)))

# def axify(i):
#     return abs(i+1) if i < 0 else i
