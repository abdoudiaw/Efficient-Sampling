#!/usr/bin/env python
#
# Author: Mike McKerns (mmckerns @uqfoundation)
# Copyright (c) 2019-2021 The Uncertainty Quantification Foundation.
# License: 3-clause BSD.  The full license text is available at:
#  - https://github.com/uqfoundation/mystic/blob/master/LICENSE

from _model import *
from ml import *
from ml import _improve_score
ar = mc.archive


def measure(xtrain, xtest, ytrain, ytest, **param): #XXX: dataset? monitor?
    """train an ANN on (x,y), with scoring calculated on (xtest,ytest),
    and return graphical distance of ANN's predictions from all data

    Input:
      x: array of x training points
      xtest: array of x test points
      y: array of y training points
      ytest: array of y test points

    Additional Input:
      pure: if True, don't use test data until final scoring
      verbose: if True, print intermediate scoring information
      delta: float, the change in target, given target is satisfied
      tries: int, number of tries to exceed target before giving up
      param: dict of hyperparameters for the MLPRegressor

    Returns:
      tuple[tuple] of:
        funcs: estimator functions (per axis)
        dist: sum of graphical distances between estimator and data (per axis)
        scores: scores for estimator (per axis)

    NOTE:
      default hyperparams are: alpha=0.0001, batch_size='auto', beta_1=0.9,
      beta_2=0.999, epsilon=1e-08, hidden_layer_sizes=(100,75,50,25),
      learning_rate_init=0.001, max_fun=15000, max_iter=1000, momentum=0.9,
      n_iter_no_change=5, power_t=0.5, tol=0.0001, validation_fraction=0.1
    """
    import sklearn.preprocessing as pre
    import sklearn.neural_network as nn

    # build dicts of hyperparameters for ANN instance
    args,barg,carg = dict(alpha=0.0001, batch_size='auto', beta_1=0.9, beta_2=0.999, epsilon=1e-08, hidden_layer_sizes=(100,75,50,25), learning_rate_init=0.001, max_fun=15000, max_iter=1000, momentum=0.9, n_iter_no_change=5, power_t=0.5, tol=0.0001, validation_fraction=0.1), dict(early_stopping=False, nesterovs_momentum=True, shuffle=True), {} #dict(activation='relu', learning_rate='constant', solver='lbfgs')
    pure = param.pop('pure', True)
    verbose = param.pop('verbose', None)
    delta = param.pop('delta', .0001)
    tries = param.pop('tries', 10)
    carg['activation'] = param.pop('activation', 'relu')
    carg['learning_rate'] = param.pop('learning_rate', 'constant')
    carg['solver'] = param.pop('solver', 'lbfgs')
    if pure:
        _xtest,xtest = xtest,xtrain
        _ytest,ytest = ytest,ytrain
    else:
        _xtest = _ytest = None

    # update hyperparameters
    args.update(param)
    mlp = nn.MLPRegressor(**args, **barg, **carg)

    ss = pre.StandardScaler()
    xscale = ss.fit_transform(xtrain)
    xstest = ss.transform(xtest)
    xyt = MLData(xscale,xstest,ytrain,ytest)

    from itertools import repeat
    imlp = repeat(mlp, shape[-1] or 1)
    iest = (Estimator(est, ss) for est in imlp)
    ixyt = repeat(xyt, shape[-1] or 1)
    vscore = lambda i,j,k: _improve_score(i,j,k, delta=delta, tries=tries, verbose=verbose, scaled=True)

    from pathos.pools import ThreadPool, ProcessPool
    #pool = ThreadPool()
    pool = ProcessPool()
    smap = pool.map
    #smap = lambda *args,**kwds: list(map(*args, **kwds))
    if shape[-1]:
        est,ypred,score_ = list(zip(*smap(vscore, range(shape[-1]), iest, ixyt)))
    else:
        est,ypred,score_ = smap(vscore, [None], iest, ixyt)[0]
    pool.close(); pool.join(); pool._clear()

    # get access to data in archive
    #data = ds.from_archive(model.__cache__(), axis=None)
    data = ds.from_archive(ar.read(fname), axis=None)

    import sklearn.metrics as sm
    if shape[-1]:
        # print results
        if pure and verbose is not False:
            print('final scores:')
            for i,si in enumerate(score_):
                print('{0}: score: {1}'.format(i,si))
        dsum,dmax = [],[]
        if pure: score_ = []
        for i,fi in enumerate(est):
            di = ds.distance(data, function=fi, axis=i)
            dsum.append(di.sum())
            dmax.append(di.max())
            # score on unseen data
            if pure: score_.append(sm.r2_score(_ytest[:,i], [fi(*x) for x in _xtest]))
        if verbose is not False:
            print('scores on test data:')
            for i,si in enumerate(score_):
                print('{0}: score: {1}, max: {2}, sum: {3}'.format(i,si,dmax[i],dsum[i]))
    else:
        # print results
        if pure and verbose is not False:
            print('final scores:')
            print('{0}: score: {1}'.format(None,score_))
        f = est
        dist = ds.distance(data, function=f, axis=None)
        dsum,dmax = (dist.sum(),dist.max())
        # score on unseen data
        if pure: score_ = sm.r2_score(_ytest, [f(*x) for x in _xtest])
        if verbose is not False:
            print('scores on test data:')
            print('{0}: score: {1}, max: {2}, sum: {3}'.format(None,score_,dmax,dsum))
    #NOTE: distance is measured against *all* data (train,test)
    return (est, dsum, score_)


if __name__ == '__main__':

    # get access to data in archive
    from mystic.monitors import Monitor
    m = Monitor()
    m._x,m._y = ds.read_archive(fname)
    xtrain,xtest,ytrain,ytest = traintest(m._x, m._y, test_size=.2, random_state=42)

    # modify MLP hyperparameters
    param = dict(alpha=0.0001, batch_size='auto', beta_1=0.9, beta_2=0.999, epsilon=1e-08, hidden_layer_sizes=(300,275,250,225,200,175,150,125), learning_rate_init=0.001, max_fun=15000, max_iter=1000, momentum=0.9, n_iter_no_change=5, power_t=0.5, tol=0.0001, validation_fraction=0.1)

    # get tuples of estimator functions, distances, and scores
    extra = dict(
        verbose = True, # if True, print intermediate scores
        pure = True, # if True, don't use test data until final scoring
        delta = .01, # step size
        tries = 10, # attempts to increase score with no improvement
        solver = 'adam',
        learning_rate = 'adaptive',
        activation = 'relu'
    #NOTE: activation: 'identity','logistic','tanh','relu'
    #NOTE: learning_rate: 'constant','invscaling','adaptive'
    #NOTE: solver: 'lbfgs','sgd','adam'
    )
    (ests,dists,scores) = measure(xtrain, xtest, ytrain, ytest, **extra, **param)

    # best (sum of) distances and scores
    #print(dists)
    #print(scores)

    # get the best estimators
    #print(ests)

    # plot the training data
    if shape[-1]:
        ypred = np.ones((xtrain.shape[0],shape[-1]))
        for i,fi in enumerate(ests):
            ypred[:,i] = [fi(*x) for x in xtrain]
    else:
        ypred = np.ones(xtrain.shape[0])
        f = ests
        ypred[:] = [f(*x) for x in xtrain]
    ax = plot_train_pred(xtrain, ytrain, ypred, xaxis=(0,1), yaxis=None, mark='oX')
    if not np.all(xtrain == xtest):
        # plot the testing data
        if shape[-1]:
            ypred = np.ones((xtest.shape[0],shape[-1]))
            for i,fi in enumerate(ests):
                ypred[:,i] = [fi(*x) for x in xtest]
        else:
            ypred = np.ones(xtest.shape[0])
            f = ests
            ypred[:] = [f(*x) for x in xtest]
        plot_train_pred(xtest, ytest, ypred, xaxis=(0,1), yaxis=None, mark=('ko','mx'), ax=ax)
    plt.show()
