
from mystic.monitors import Monitor, LoggingMonitor
import dataset as ds
import numpy as np
import plotter as plt
import interpolator as itp

m = Monitor()
mname = 'demo'  # dynamic 'model' name
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

print(time.now())


