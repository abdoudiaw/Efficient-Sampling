from mystic.monitors import LoggingMonitor
import numpy as np

mon = LoggingMonitor(filename ='MD_data.csv')


data=np.loadtxt('rdf.csv', skiprows=0, unpack=True)
Gamma= np.full((len(data[1])), 100, dtype=float)

x=[i for i in zip(Gamma,data[1])]
y=data[2]

z = zip(x,y)
import mystic.cache as mc
db = mc.archive.read('foo')
mc.archive.write(db, z)
