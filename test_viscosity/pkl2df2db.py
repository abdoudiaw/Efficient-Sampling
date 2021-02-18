import pandas as pd
import potatoe as rb

from _model import *
cols = ['TEMPERATURE', 'DENSITY_0', 'DENSITY_1', 'CHARGES_0', 'CHARGES_1', 'DIFFCOEFF_0', 'DIFFCOEFF_1', 'DIFFCOEFF_4']

# read pickled data into dataframe
df = pd.read_pickle('lmp_dataframe.pkl')
df = df[cols]
df.loc[:,cols[1]] = df.loc[:,cols[1]]/1e23
df.loc[:,cols[2]] = df.loc[:,cols[2]]/1e23

if shape[0] == 5:
  inputs = cols[:5]
elif shape[0] == 3:
  inputs = [cols[0]] + cols[1:5:2]
elif shape[0] == 1:
  inputs = [cols[0]]
else:
  raise NotImplementedError('shape[0] must be in [5,3,1]')
if shape[1] == 3:
  output = cols[5:]
elif shape[1] == 2:
  output = cols[5:-1]
elif shape[1] == 1:
  output = cols[5:-2]
elif shape[1] == 0:
  output = cols[5]
else:
  raise NotImplementedError('shape[1] must be in [3,2,1]')

# write entries to database
if shape[1]:
    entries = dict((tuple(i.tolist()),tuple(j.tolist())) for i,j in zip(df[inputs].values, df[output].values))
else:
    entries = dict((tuple(i.tolist()),j.tolist()) for i,j in zip(df[inputs].values, df[output].values))
db = rb.read(fname)
rb.write(db, entries)
