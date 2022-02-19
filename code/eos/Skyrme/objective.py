import torch
import collections
import sys
torch.set_default_dtype(torch.float64)
sys.path.insert(0, 'skyrme_model')

import nn_learner


def presssure(x):
    """Skyrme bag hybrid model
    
    Inputs:
      x: proton fraction and density
    Output:
        z: pressure
    """
    YP, NB = x
    Inputs = collections.namedtuple('Inputs', 'YP NB')
    model = torch.load("./skyrme_model/skyrme_bag_hybrid.pt")
    return model(Inputs(YP, NB))[0][0]
