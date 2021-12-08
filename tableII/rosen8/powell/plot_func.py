# check interpolated model
import mystic as my
import mystic.cache as mc
from _model import hcube, ename, mname, scale

# get learned model
try:
  f = mc.archive.read_func('func.db', n=0)[0]
  my.model_plotter(lambda x: f(*x), depth=True, bounds=hcube, scale=scale)
except:
  pass

# check truth model
import dill
objective = dill.load(open('cost.pkl', 'rb'))
my.model_plotter(objective, depth=True, bounds=hcube, scale=scale)

# get sampled points
data = mc.archive.read(mname)
my.model_plotter(objective, data, depth=False, bounds=hcube, dots=True, join=False)

# get terminated points
data = mc.archive.read(ename)
my.model_plotter(objective, data, depth=False, bounds=hcube, dots=True, join=False)
