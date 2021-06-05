# check interpolate model
import mystic as my
import mystic.cache as mc
from _model import upper, ename
lb, ub, st = 0, upper, upper/50.
bounds = '{0}:{1}:{2}, {0}:{1}:{2}'.format(lb, ub, st)
bounds = "0:10:0.2, 0:10:0.2"
#bounds += ', 0.0'
# get learned model
try:
  #f = mc.function.read('func.db') #FIXME: n
  f = mc.archive.read_func('func.db', n=0)[0]
  my.model_plotter(lambda x: f(*x), depth=True, bounds=bounds)
except:
  pass
# check truth model
from _model import objective
import dill
objective = dill.load(open('cost.pkl', 'rb'))
my.model_plotter(objective, depth=True, bounds=bounds)

# get terminated points
extrema = mc.archive.read(ename)
em = my.monitors.Monitor()
if len(extrema):
  em._x, em._y = zip(*extrema.items())
  em._y = list(em._y)
  em._x = [list(x) for x in em._x]

my.model_plotter(objective, em, depth=False, bounds=bounds, dots=True, join=False)
