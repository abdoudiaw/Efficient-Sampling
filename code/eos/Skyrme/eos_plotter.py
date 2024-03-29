#!/usr/bin/env python
#
# Author: Mike McKerns (mmckerns @uqfoundation)
# Copyright (c) 2018-2019 The Uncertainty Quantification Foundation.
# License: 3-clause BSD.  The full license text is available at:
#  - https://github.com/uqfoundation/mystic/blob/master/LICENSE
"""
plotter for data (x,z) and response surface function(*x)
  - initalize with x and z (and function)
  - interpolate if function is not provided
  - can downsample
  - plot data and response surface
"""

class Plotter(object):

    def __init__(self, x, z=None, function=None, **kwds):
        """scatter plotter for data (x,z) and response surface function(*x)

        Input:
          x: an array of shape (npts, dim) or (npts,)
          z: an array of shape (npts,)
          function: function f, where z=f(*x.T), or str (interpolation method)

        Additional Inputs:
          step: int, plot every 'step' points on the grid [default: 200]
          scale: float, scaling factor for the z-axis [default: False]
          shift: float, additive shift for the z-axis [default: False]
          density: int, density of wireframe for the plot surface [default: 9]
          axes: tuple, indicies of the axes to plot [default: ()]
          vals: list of values (one per axis) for unplotted axes [default: ()]
          maxpts: int, maximum number of (x,z) points to use [default: None]
          kernel: function transforming x to x', where x' = kernel(x)

        NOTE:
          if scipy is not installed, will use np.interp for 1D (non-rbf),
          or mystic's rbf otherwise. default method is 'nearest' for
          1D and 'linear' otherwise. method can be one of ('rbf','linear',
          'nearest','cubic','inverse','gaussian','quintic','thin_plate').
        """
        self.x = getattr(x, '_x', x)  # params (x)
        self.z = x._y if z is None else z # cost (f(x))
        if function is None:
            function='linear'
        if type(function) is str:
            from mystic.math.interpolate import interpf
            function = interpf(self.x,self.z, method=function, arrays=True) #XXX: kwds?
        self.function = function
       #self.dim = kwds.pop('dim', None) #XXX: or len(x)?
        # interpolator configuration
        self.args = dict(step=200, scale=False, shift=True, \
            kernel=None, density=9, axes=(), vals=(), maxpts=None)
        self.args.update(kwds)
        self.maxpts = self.args.pop('maxpts')
        return

    def _downsample(self, maxpts=None, x=None, z=None):
        """downsample (x,z) to at most maxpts

        Input:
          maxpts: int, maximum number of points to use from (x,z)
          x: an array of shape (npts, dim) or (npts,)
          z: an array of shape (npts,)

        Output:
          x: an array of shape (npts, dim) or (npts,)
          z: an array of shape (npts,)
        """
        if maxpts is None: maxpts = self.maxpts
        if x is None: x = self.x
        if z is None: z = self.z
        if len(x) != len(z):
            raise ValueError("the input array lengths must match exactly")
        if maxpts is not None and len(z) > maxpts:
            N = max(int(round(len(z)/float(maxpts))),1)
        #   print("for speed, sampling {} down to {}".format(len(z),len(z)/N))
        #   ax.plot(x[:,0], x[:,1], z, 'ko', linewidth=2, markersize=4)
            x = x[::N]
            z = z[::N]
        #   plt.show()
        #   exit()
        return x, z

    def _max(self):
        """get the x[i],z[i] corresponding to the max(z)
        """
        import numpy as np
        mz = np.argmax(self.z)
        return self.x[mz], self.z[mz]

    def _min(self):
        """get the x[i],z[i] corresponding to the min(z)
        """
        import numpy as np
        mz = np.argmin(self.z)
        return self.x[mz], self.z[mz]

    def Plot(self, **kwds):
        """produce a scatterplot of (x,z) and the surface z = function(*x.T)

        Input:
          step: int, plot every 'step' points on the grid [default: 200]
          scale: float, scaling factor for the z-axis [default: False]
          shift: float, additive shift for the z-axis [default: False]
          density: int, density of wireframe for the plot surface [default: 9]
          axes: tuple, indicies of the axes to plot [default: ()]
          vals: list of values (one per axis) for unplotted axes [default: ()]
          maxpts: int, maximum number of (x,z) points to use [default: None]
          kernel: function transforming x to x', where x' = kernel(x)
        """
        step = kwds['step'] if 'step' in kwds else self.args['step']
        scale = kwds['scale'] if 'scale' in kwds else self.args['scale']
        shift = kwds['shift'] if 'shift' in kwds else self.args['shift']
        axes = kwds['axes'] if 'axes' in kwds else self.args['axes']
        vals = kwds['vals'] if 'vals' in kwds else self.args['vals']
        maxpts = kwds['maxpts'] if 'maxpts' in kwds else self.maxpts
        kernel = kwds['kernel'] if 'kernel' in kwds else self.args['kernel']
        density = kwds['density'] if 'density' in kwds else self.args['density']

        # plot response surface
        from mpl_toolkits.mplot3d import axes3d
        import matplotlib.pyplot as plt
        from matplotlib import cm
        import numpy as np

        figure = plt.figure(figsize = (10, 8))
        kwds = {'projection':'3d'}
#        fig, ax= plt.subplots(figsize = (15, 10))
        ax = figure.gca(**kwds)
        ax.autoscale(tight=False)

        x, z = self._downsample(maxpts)
        x = np.asarray(x)

        # get two axes to plot, and indices of the remaining axes
        axes = axes[:2]  #XXX: error if wrong size?
        ix = [i for i in range(len(x.T)) if i not in axes]
        n = 2-len(axes)
        axes, ix = list(axes)+ix[:n], ix[n:]

        # build list of fixed values (default mins), override with user input
       #fix = np.zeros(len(ix))
        fix = enumerate(self._min()[0])
        fix = np.array(tuple(j for (i,j) in fix if i not in axes))
        fix[:len(vals)] = vals

        # build grid of points, one for each param, apply fixed values
        grid = np.ones((len(x.T),step,step))
        grid[ix] = fix[:,None,None]
        del ix, fix

        # build sub-surface of function(x) to display, apply to the grid
        xy = x.T[axes]
        M = complex('{}j'.format(step))
        grid[axes] = np.mgrid[xy[0].min():xy[0].max():M,
                              xy[1].min():xy[1].max():M]
        del xy

        # evaluate the function on the sub-surface
        z_ = np.asarray(self.function(*grid))
        # scaling used by function plotter
        print(scale)
        if scale:
            if shift:
                z_ = z_+shift
            z_ = np.log(4*z_*scale+1)+2

        # apply transform #NOTE: should do this w/o fixing points first
        if hasattr(kernel, '__call__'):
            _grid = np.zeros_like(grid[:2])
            for i in range(step):
                _grid.T[i] = [kernel(j)[:2] for j in grid.T[i]] #XXX: correct?
            grid = _grid
            ax0,ax1 = 0,1
        else: ax0,ax1 = axes

        # plot surface
       
        d = max(11 - density, 1)
        d = max(11 - density, 1)
        print("d",d)
        x_ = grid[ax0]
        y_ = grid[ax1]
        ax.xaxis.set_tick_params(labelsize=20)
        ax.yaxis.set_tick_params(labelsize=20)
        ax.zaxis.set_tick_params(labelsize=20)

        ax.plot_wireframe(x_, y_, z_, rstride=d, cstride=d, color = 'green',linewidth = 1,
                          antialiased = True)
        ax.set_zlabel('pressure [MeV f m$^{-3}$]',fontsize=20, labelpad=30)
        ax.set_xlabel('proton fraction',fontsize=20, labelpad=20)
        ax.set_ylabel('density [f m$^{-3}$]',fontsize=20, labelpad=20)
        ax.set_yticks([0.,0.4,0.8,1.2,1.6])
        ax.set_xticks([0.,0.2,0.4,0.6])
        ax.tick_params(axis='z', pad=10)
   

        ax.grid(False)
        ax.view_init(elev=20, azim=-125)
        
#        ax.plot_surface(x_, y_, z_, rstride=d, cstride=d, cmap=cm.plasma, linewidth=0, antialiased=False)

        # use the sampled values
        z_ = np.asarray(z)
        # scaling used by function plotter
        if scale:
            if shift:
                z_ = z_+shift
            z_ = np.log(4*z_*scale+1)+2

        # apply transform
        if hasattr(kernel, '__call__'):
            x = np.array([kernel(j)[:2] for j in x])

        # plot data points
        x_ = x.T[ax0]
        y_ = x.T[ax1]
        ax.plot(x_, y_, z_, 'bo', linewidth=2, markersize=2.5)


#        plt.savefig('eos_surf_plot.png',dpi=199, bbox_inches='tight')
        plt.tight_layout()
        plt.show()  #XXX: show or don't show?... or return?


def plot(monitor, function=None, **kwds):
    '''generic interface to Plotter, returning an Plotter instance

    Input:
      monitor: a mystic.monitor instance
      function: function f, where z=f(*x.T), or str (interpolation method)

    Additional Inputs:
      step: int, plot every 'step' points on the grid [default: 200]
      scale: float, scaling factor for the z-axis [default: False]
      shift: float, additive shift for the z-axis [default: False]
      density: int, density of wireframe for the plot surface [default: 9]
      axes: tuple, indicies of the axes to plot [default: ()]
      vals: list of values (one per axis) for unplotted axes [default: ()]
      maxpts: int, maximum number of (x,z) points to use [default: None]
      kernel: function transforming x to x', where x' = kernel(x)

    NOTE:
      if scipy is not installed, will use np.interp for 1D (non-rbf),
      or mystic's rbf otherwise. default method is 'nearest' for
      1D and 'linear' otherwise. method can be one of ('rbf','linear',
      'nearest','cubic','inverse','gaussian','quintic','thin_plate').
    '''
    p = Plotter(monitor, function=function, **kwds)
    p.Plot()
    return p  #XXX: return nothing?


# EOF
