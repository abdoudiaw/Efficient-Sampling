"""
0) if no interpf, then launch searchers
1) then (when?) interpolate
2) if interpf meets quality metric, then stop searching [per axis]
3) upon new data, check interpf result vs data
4) if interpf fails quality metric, then (re-)interpolate
5) if interpf fails quality metric, then (re-)launch searchers
6) repeat from step #1
"""
from _prep import *

'''
#FIXME: run searchers in parallel
#FIXME: internalize cache, inversion, ...?
if shape[-1]:
    for i in range(shape[-1]):
        s = sample(i)
        print('{i}: evals:{e}, cache:{c}'.format(i=i,e=s.evals(),c=len(model.__cache__())))
    for i in range(shape[-1]):
        s = isample(i)
        print('-{i}: evals:{e}, cache:{c}'.format(i=i,e=s.evals(),c=len(model.__cache__())))
else:
    i = '*'
    s = sample()
    print('{i}: evals:{e}, cache:{c}'.format(i=i,e=s.evals(),c=len(model.__cache__())))
    s = isample()
    print('-{i}: evals:{e}, cache:{c}'.format(i=i,e=s.evals(),c=len(model.__cache__())))
'''

def valid(dist, max=1e-4, sum=1e-2):
    """True if max(dist) <= max and sum(dist) <= sum

    dist: numpy array of distances of shape (npoints,)
    max: float, largest acceptable distance
    sum: float, largest acceptable sum of all distances
    """
    return dist.max() <= max and dist.sum() <= sum

"""
0) start:
     if given func: go to (4)
     elif given data (and no func), go to (1)
     else: launch search in each new dimension
1) (search is complete [on axis X]), interpf [on axis X]
2) if distance(data, interpf[axis], axis) > tol:
     launch new searcher, repeat from (1)
3) save interpf [on axis X] (and dist, and id)
4) new data: if distance(data, interpf[axis], axis) > tol (stored f)
     stop
   else interpf [on axis X]
5) repeat from (2)
"""

#NOTE: this is one axis, should run for each axis (both model & imodel)
#NOTE: works for data,func single-valued and multi-valued?
#NOTE: store func (and dist & id) [can pickle func.__axis__, not but func]
#FIXME: either enable pickle of func, or rebuild func from func.__axis__
#FIXME: save and read func from DB, handle rerun with existing func/run DBs
#XXX: enable use of interpf kwds (e.g. smooth) (in interpf, not interpf_nd)
#XXX: enable search parallel run (has map, but is not used)
def validate(x, z, axis, data=None, func=None, **kwds):
    """ensure function of (x,z) is valid with respect to all data

    Inputs:
      x: an array of shape (npts, dim) or (npts,)
      z: an array of shape (npts, N) or (npts,)
      axis: int in [0,N], the axis of z to select
      data: a mystic.math.legacydata.dataset of legacy data
      func: interpolated function z = f(*x) for data (x,z)

    Additional Inputs:
      warm: int, search until "warm" samples are taken (default is 0)
      iters: int, abort after iters (default is inf)
      retain: True if retain (x,z) in model cache (default is True)

    Output:
      returns a tuple of (interpolated function, graphical distance)

    NOTE:
      additional keyword arguments are avaiable for interpolation. See
      mystic.math.interpolate.interpf for more details.
    """ # data,func single-valued
    warm = kwds.get('warm', 0)
    iters = kwds.get('iters', float('inf'))
    retain = kwds.get('retain', True)
    _tuple = lambda i: (tuple(i) if shape[-1] else i)
    # get archived model evaluations
    c = model.__cache__()
    #ax = axify(axis)
    import dataset as ds
    import mystic.math.legacydata as ld
    from mystic.math.interpolate import _unique as unique
    # include desired points in cache
    if retain:
        x,z = unique(x,z)
        x,z = x.tolist(),[_tuple(i) for i in z]
        for xi,zi in zip(x,z):
            c.update({tuple(xi):zi}) #XXX: assumes 'clear' key,value
    elif retain is None: #XXX: retain, but reevaluate z #FIXME: easter egg
        x = unique(x).tolist()
        z = [model(i) for i in x]
    else: # temporarily include (x,z) in data
        x,z = unique(x,z) #XXX: or pass?
        x,z = x.tolist(),[_tuple(i) for i in z]
    # add new data points to the data
    if data is None:
        data = ld.dataset()
    elif data is True: #FIXME: easter egg
        data = ds.from_archive(c, axis=None)  
    else:
        data = ld.dataset().load(data.coords, data.values) #XXX: new ids
    data = data.load(x,z) # ids?
    xx,zz = unique(data.coords, data.values)
    data = ld.dataset().load(xx.tolist(),[_tuple(i) for i in zz]) #XXX: new ids
    del xx,zz
    # check for validity of the function
    if func is None:
        import numpy as np
        import interpolator as itp
        from mystic.monitors import Monitor
        dist = np.array([np.inf]) # initialize as False
        repeat = iters
        while not valid(dist) and repeat:
            # interpolate
            m = Monitor()
            m._x,m._y = data.coords,data.values
            f = itp.Interpolator(m, **kwds).Interpolate(axis=axis)
            # calculate distance/validity
            dist = ds.distance(data, function=f, axis=axis)
            warm_ = not max(0, warm-len(c))
            if valid(dist) and warm_: #XXX: will this always be valid?
                break
            msg = 'warming' if valid(dist) else 'invalid'
            print('{msg}: max:{max}, sum:{sum}'.format(msg=msg, max=dist.max(), sum=dist.sum()))
            # launch searchers #XXX: use (x,z) to seed the search?
            #i = '+' if axis == ax else '-'
            #print('{i}{a}: evals:{e}, cache:{c}'.format(i=i,a=ax,e=s.evals(),c=len(c)))
            s = sum(s.evals() for s in search(axis)) #XXX: Pool().map?
            
            print('{a}: evals:{e}, cache:{c}'.format(a=axis,e=s,c=len(c)))
            data = ds.from_archive(c, axis=None).load(x,z) #XXX: don't load?
            dist = ds.distance(data, function=f, axis=axis)
            repeat -= 1
        if valid(dist):
            print('valid: max:{max}, sum:{sum}'.format(max=dist.max(), sum=dist.sum()))
        else:
            print('quit: max:{max}, sum:{sum}'.format(max=dist.max(), sum=dist.sum()))
        return f, dist
    else:
        # calculate distance/validity
        dist = ds.distance(data, function=func, axis=axis)
        if not valid(dist):
            # recalculate with new function
            print('invalid: max:{max}, sum:{sum}'.format(max=dist.max(), sum=dist.sum()))
            return validate(x,z, axis=axis, data=data, func=None, **kwds)
        print('valid: max:{max}, sum:{sum}'.format(max=dist.max(), sum=dist.sum()))
        return func, dist

"""
pts = 2
import numpy as np
from interpf import interpf_nd
#x = [[1,2,3,4,5],[6,7,8,9,10]]
x = np.arange(1,1+pts*shape[0]).reshape(pts,-1).tolist()
z = [objective(i) for i in x]
if shape[-1]:
    # generate a dummy interpf to store results
    func = interpf_nd([[1,2],[2,3]],[[1,2],[2,3]],method='thin_plate')
    func.__axis__[:] = [validate(x, z, axis=i, data=None, func=None, method='thin_plate')[0] for i in range(shape[-1])]
else:
    func = validate(x, z, axis=None, data=None, func=None, method='thin_plate')[0]
    func.__axis__ = None
data = ds.from_archive(model.__cache__(), axis=None)#.load(x, z)

#x = [[11,12,13,14,15],[16,17,18,19,20]]
x = np.array(x); x = (x + x.size).tolist()
z = [objective(i) for i in x]
if shape[-1]:
    func.__axis__[:] = [validate(x, z, axis=i, data=data, func=func.__axis__[i], method='thin_plate')[0] for i in range(shape[-1])]
else:
    func = validate(x, z, axis=None, data=data, func=func, method='thin_plate')[0]
    func.__axis__ = None
data = ds.from_archive(model.__cache__(), axis=None)#.load(x, z)

#x = [[111,112,113,114,115],[116,117,118,119,120]]
x = np.array(x); x = (x + x.size**2).tolist()
z = [objective(i) for i in x]
if shape[-1]:
    func.__axis__[:] = [validate(x, z, axis=i, data=data, func=func.__axis__[i], method='thin_plate')[0] for i in range(shape[-1])]
else:
    func = validate(x, z, axis=None, data=data, func=func, method='thin_plate')[0]
    func.__axis__ = None
data = ds.from_archive(model.__cache__(), axis=None)#.load(x, z)
"""
from _potatoe import _func_db, read_func, write_func

#'''
pts = 20
import numpy as np
print('new data points (x,z)')
#x = np.random.rand(pts,shape[0]).tolist() #XXX: slight chance of duplicates
x = [(0.1,0.1),(1,1)]
z = [objective(i) for i in x]

# get handles to func_DBs
if shape[-1]:
    archives = list(map(lambda i: _func_db('func{i}.db'.format(i=i)), range(shape[-1])))
else:
    archives = _func_db('func.db')

# check for stored func in func_db, and if not found
# generate a dummy interpf to store results
func = read_func(archives)
'''
# generate a dummy interpf to store results #FIXME: shape[-1] == 0?
from interpf import interpf_nd
func = interpf_nd([[1,2],[2,3]],[[1,2],[2,3]],method='thin_plate')
func.__axis__[:] = [None]*shape[-1]
'''
d = [None]
if shape[-1]:
    do = lambda : (lambda i: d[i])(0)
    xyz = lambda i: validate(x, z, axis=i, data=do(), func=func.__axis__[i], warm=1000, method='thin_plate') #XXX: epsilon?
else:
    do = lambda : (lambda i: d[0])(0)
    xyz = lambda i: validate(x, z, axis=None, data=do(), func=func, warm=1000, method='thin_plate') #XXX: epsilon?

def xxx(i):
    _f,_d = xyz(i)
    # read data from (cached) run archive
    if (shape[-1] and None in func.__axis__) or (func is None):
        print('{i}: no stored function'.format(i=i))
        return (_f,_d)[0]
    import dataset as ds
    data = ds.from_archive(model.__cache__(), axis=None) #XXX: axis?
    # calculate dist from data to func
    dist = ds.distance(data, function=func, axis=None) #XXX: axis?
    # keep the func with the smaller distance
    _d, d = _d.sum(), dist[i].sum()
    if _d < d:
        print('{i}: replace stored function'.format(i=i))
        return (_f,_d)[0]
    print('{i}: retain stored function'.format(i=i))
    return (func.__axis__[i],d)[0] if shape[-1] else (func,d)[0]

import multiprocess as mp
pool = mp.Pool()
smap = pool.map
#smap = lambda *args,**kwds: list(map(*args, **kwds))

print('force sampling in process-parallel')
if shape[-1]:
    func.__axis__[:] = smap(xxx, range(shape[-1]))  # data = None, func = None
else:
    func = smap(xxx, [None])[0]
write_func(func, archives)
pool.close(); pool.join()
d[:] = [True]
import multiprocess.dummy as mt
pool = mt.Pool()
smap = pool.map
#smap = lambda *args,**kwds: list(map(*args, **kwds))

print('interpolate given data in thread-parallel')
if shape[-1]:
    func.__axis__[:] = smap(xxx, range(shape[-1]))  # data and func provided
else:
    func = smap(xxx, [None])[0]
write_func(func, archives)
pool.close(); pool.join()
import dataset as ds
data = ds.from_archive(model.__cache__(), axis=None)
for f in (func.__axis__ if shape[-1] else [func]):
    dist = ds.distance(data, function=f)#, axis=None)
    assert valid(dist)
    print('valid: OK')

print('new data points (x,z)')
#x = np.random.rand(pts,shape[0]).tolist() #XXX: slight chance of duplicates

x = [(0.85,0.95),(.01,.9)]
z = [objective(i) for i in x]
smap = lambda *args,**kwds: list(map(*args, **kwds))

print('interpolate given data in not-parallel')
if shape[-1]:
    func.__axis__[:] = smap(xxx, range(shape[-1]))  # data and func provided
else:
    func = smap(xxx, [None])[0]
write_func(func, archives)
data = ds.from_archive(model.__cache__(), axis=None)
for f in (func.__axis__ if shape[-1] else [func]):
    dist = ds.distance(data, function=f)#, axis=None)
    assert valid(dist)
    print('valid: OK')

#'''

#'''
from mystic.monitors import Monitor, LoggingMonitor
import dataset as ds
import numpy as np
import plotter as plt
import interpolator as itp

m = Monitor()
m._x,m._y = ds.read_archive(mname)

# if don't use dummy 'func' above
r = itp.Interpolator(m, method='thin_plate')
func = r.Interpolate()

print('Number of function evaluation', len(m._x))

print('plot surface generated with interpolated function')


p = plt.Plotter(m, function=func)
if shape[-1]:
    for i in range(shape[-1]):
        p.Plot(axis=i)
else:
    p.Plot()
#'''
