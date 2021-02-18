def _getaxis(z, axis):
    """get the selected axis of the multi-valued array

    Inputs:
      z: an array of shape (npts, N)
      axis: int, the desired index the multi-valued array [0,N]
    """
    if len(z) and not hasattr(z[0], '__len__'):
        msg = "cannot get axis=%s for single-valued array" % axis
        raise ValueError(msg)
    if axis is None:
        axis = slice(None)
    elif len(z) and axis >= len(z[0]):
        if len(z[0]) < 1:
            msg = "cannot get axis=%s for empty array" % axis #XXX: ?
        else:
            msg = "axis should be an int in the range 0 to %s" % (len(z[0])-1)
        raise ValueError(msg)
    # select values corresponding to axis
    if type(z) not in (list, tuple):
        return z[:,axis]
    return type(z)(i[axis] for i in z)


def interpf_nd(x, z, method=None, extrap=False, arrays=False, **kwds):
    '''interpolate (x,z) to generate f, where z=f(*x)

    Input:
      x: an array of shape (npts, dim) or (npts,)
      z: an array of shape (npts, N)
      method: string for kind of interpolator
      extrap: if True, extrapolate a bounding box (can reduce # of nans)
      arrays: if True, z = f(*x) is a numpy array; otherwise don't use arrays
      axis: int in [0,N], the axis of z to interpolate (all, by default)
      map: map function, used to parallelize interpf for each axis

    Output:
      interpolated function f, where z=f(*x)

    NOTE:
      if scipy is not installed, will use np.interp for 1D (non-rbf),
      or mystic's rbf otherwise. default method is 'nearest' for
      1D and 'linear' otherwise. method can be one of ('rbf','linear','cubic',
      'nearest','inverse','gaussian','multiquadric','quintic','thin_plate').

    NOTE:
      if extrap is True, extrapolate using interpf with method='thin_plate'
      (or 'rbf' if scipy is not found). Alternately, any one of ('rbf',
      'linear','cubic','nearest','inverse','gaussian','multiquadric',
      'quintic','thin_plate') can be used. If extrap is a cost function
      z = f(x), then directly use it in the extrapolation.

    NOTE:
      additional keyword arguments (epsilon, smooth, norm) are avaiable
      for use with a Rbf interpolator. See mystic.math.interpolate.Rbf
      for more details.
    '''
    axis = kwds.get('axis', None)
    _map = kwds.get('map', map)
    _kwd = dict(method=method, extrap=extrap, arrays=arrays)
    if 'norm' in kwds: _kwd['norm'] = kwds.pop('norm')
    if 'smooth' in kwds: _kwd['smooth'] = kwds.pop('smooth')
    if 'epsilon' in kwds: _kwd['epsilon'] = kwds.pop('epsilon')
    # interpolate for each member of tuple-valued data, unless axis provided
    if axis is None:
        if len(z) and hasattr(z[0], '__len__'):
            #zt = type(z[0])
            import numpy as np
            #if zt is np.ndarray: zt = np.array
            zt = np.array if arrays else list #XXX: tuple(zt(*xi)) or zt(*x) ?
            # iterate over each axis, build a 'combined' interpf
            def function(*args, **kwds): #XXX: z = array(f(*x.T)).T
                axis = kwds.get('axis', None)
                fs = function.__axis__
                if axis is None:
                    if hasattr(args[0], '__len__'):
                        return tuple(zt(fi(*args)) for fi in fs)
                    return tuple(fi(*args) for fi in fs)
                return fs[axis](*args)
            def interpf_ax(i):
                return interpf_nd(x, z, axis=i, **_kwd)
            function.__axis__ = list(_map(interpf_ax, range(len(z[0]))))
            #function.__axis__ = [interpf_nd(x, z, axis=ax, **_kwd) for ax,val in enumerate(z[0])]
            return function
    else:
        z = _getaxis(z, axis)
    #XXX: what if dataset is empty? (i.e. len(data.values) == 0)
    #NOTE: the following is the same as Interpolator(...)._interpolate(...)
    import numpy as np
    from mystic.math.interpolate import interpf
    with np.warnings.catch_warnings():
        np.warnings.filterwarnings('ignore')
        function = interpf(x, z, **_kwd)
    # from mystic.math.interpolate import _to_objective
    # function = _to_objective(function)
    function.__axis__ = axis #XXX: bad idea: list of funcs, or int/None ?
    return function

