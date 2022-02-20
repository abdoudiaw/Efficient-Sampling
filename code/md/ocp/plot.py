
"""
0) if no interpf, then launch searchers
1) then (when?) interpolate
2) if interpf meets quality metric, then stop searching [per axis]
3) upon new data, check interpf result vs data
4) if interpf fails quality metric, then (re-)interpolate
5) if interpf fails quality metric, then (re-)launch searchers
6) repeat from step #1
"""
from prep import *

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

print('plot surface generated with interpolated function')
p = plt.Plotter(m, function=func)
if shape[-1]:
    for i in range(shape[-1]):
        p.Plot(axis=i)
else:
    p.Plot()
