#!/usr/bin/env python
#
# Author: Mike McKerns (mmckerns @uqfoundation)
# Copyright (c) 2019-2021 The Uncertainty Quantification Foundation.
# License: 3-clause BSD.  The full license text is available at:
#  - https://github.com/uqfoundation/mystic/blob/master/LICENSE

from _mlearn import *
import mystic.constraints as co
rb = mc.function


if __name__ == '__main__':

    # get access to data in archive
    from mystic.monitors import Monitor
    m = Monitor()
    m._x,m._y = ds.read_archive(pname)
    xyt = traintest(m._x, m._y, test_size=.2, random_state=42)

    # get handles to func_DBs
    if shape[-1]:
        archives = list(map(lambda i: rb.db('est{i}.db'.format(i=i)), range(shape[-1])))
    else:
        archives = rb.db('est.db')

    # check for stored func in func_db, and if not found
    # generate a dummy estimator to store results
    func = rb.read(archives)

    xtrain,xtest,ytrain,ytest = xyt

    # plot the training data
    if shape[-1]:
        ypred = np.ones((xtrain.shape[0],shape[-1]))
        for i,fi in enumerate(func.__axis__):
            ypred[:,i] = [fi(*x) for x in xtrain]
    else:
        ypred = np.ones(xtrain.shape[0])
        ypred[:] = [func(*x) for x in xtrain]
    ax = plot_train_pred(xtrain, ytrain, ypred, xaxis=(0,1), yaxis=None, mark='oX')

    if not np.all(xtrain == xtest):
        # plot the testing data
        if shape[-1]:
            ypred = np.ones((xtest.shape[0],shape[-1]))
            for i,fi in enumerate(func.__axis__):
                ypred[:,i] = [fi(*x) for x in xtest]
        else:
            ypred = np.ones(xtest.shape[0])
            ypred[:] = [func(*x) for x in xtest]
        plot_train_pred(xtest, ytest, ypred, xaxis=(0,1), yaxis=None, mark=('ko','mx'), ax=ax)
    plt.show()
