
from _model import *

# apply cache/archive in 3D same as 1D
import klepto as kl
import mystic as my

cache = kl.inf_cache(cache=kl.archives.dir_archive(mname, cached=False), keymap=kl.keymaps.keymap(), ignore='**')
inner = cache(lambda *x, **kwds: objective(x, axis=kwds.get('axis', None)))
_model = lambda x, axis=None: inner(*x, **dict(axis=axis))
_model.__inner__ = inner

# when caching, always cache returned tuple (_model caches based on 'axis')
if shape[-1]:
    def model(x, axis=None):
        if axis is None: axis = slice(None)
        return _model(x)[axis]
else: # z is single-valued
    def model(x, axis=None): # ignore axis
        return _model(x)

model.__cache__ = lambda : inner.__cache__()

# get "model inverse" (for maximization)
imodel = lambda *args, **kwds: -model(*args, **kwds)

model.__inverse__ = imodel
imodel.__inverse__ = model
imodel.__cache__ = model.__cache__

#XXX (apply logger?)

def sample(axis=None, invert=False):
    """search (for minima) until terminated
    """#FIXME: {solver:directed or None}, {terminated:all, evals:1000}
    _model = imodel if invert else model
    if axis is None:
        model_ = lambda x: _model(x)
    else:
        model_ = lambda x: _model(x, axis)
    s = searcher(bounds, model_, npts=npts, solver=solver)
    s.sample_until(terminated=all)
#    s.sample_until(iters=100)

#    s.sample_until(evals=100)
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
