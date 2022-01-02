import numpy as np
import mystic as my
import matplotlib.pyplot as plt
try:
    x,y = my.munge.read_history('tightopt/score.txt')
    x = np.array(x).flatten().tolist()
    plt.plot(x,y)
except: pass
try:
    xx,yy = my.munge.read_history('tightrnd/score.txt')
    xx = np.array(xx).flatten().tolist()
    plt.plot(xx,yy)
except: pass
plt.legend(['optimizer-directed', 'random'])
plt.xlabel('function evaluations')
plt.ylabel('test score')
plt.show()
