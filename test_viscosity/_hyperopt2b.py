'''
enable hyperparameter optimization for code in _mlearn
'''
from _mlearn2b import *
from _potatoe import _func_db, read_func, write_func


def hypercost(p, x,xt,y,yt): #XXX: ExtraArgs or factory?
    # learning_rate_init = p[0]  # initial learning rate
    # beta_1 = p[1]              # decay rate for 1st moment vector
    # beta_2 = p[2]              # decay rate for 2nd moment vector
    #NOTE: other hyperparameters need solver ('sgd')
    learning, beta1, beta2 = p

    # modify MLP hyperparameters
    param = dict(beta_1=beta1, beta_2=beta2, learning_rate_init=learning, hidden_layer_sizes=(200,175,150,125), pure=True, verbose=True, delta=.01, tries=2) #FIXME: delta=.1,tries=1 for demo only
    extra = dict(activation='relu', solver='adam', learning_rate='constant') #FIXME: identity for demo only
    (funcs,dists,scores) = measure(x, xt, y, yt, **param, **extra)

    #""" #NOTE: faster w/o calculate dist and save function
    # get handles to func_DBs
    if shape[-1]:
        archives = list(map(lambda i: _func_db('est{i}.db'.format(i=i)), range(shape[-1])))
    else:
        archives = _func_db('est.db')

    # check for stored func in func_db, and if not found
    # generate a dummy estimator to store results
    func = read_func(archives)

    smap = lambda *args,**kwds: list(map(*args, **kwds))

    def xxx(i): #XXX: duplicates much of _workflow.xxx
        if i is None:
            _f,_d = funcs,dists # new results
        else:
            _f,_d = funcs[i],dists[i] # new results
        if (shape[-1] and None in func.__axis__) or (func is None):
            return (_f,_d,True)
        import dataset as ds
        #data = ds.from_archive(model.__cache__(), axis=None)
        data = ds.from_archive(ds.load_archive(fname), axis=None)
        dist = ds.distance(data, function=func, axis=None)
        dist = dist.sum() if i is None else dist[i].sum()
        # keep the func with the smaller distance
        if _d < dist: # new is smaller
            return (_f,_d,True)
        return ((func.__axis__[i] if shape[-1] else func),dist,False)

    if shape[-1]:
        _funcs,_dists,_new = list(zip(*smap(xxx, range(shape[-1]))))
    else:
        _funcs,_dists,_new = smap(xxx, [None])[0]

    if np.any(_new): #XXX: is a 'worse' component ever written?
        if shape[-1]:
            func.__axis__[:] = _funcs
        else:
            func = _funcs
        write_func(func, archives) #FIXME: need save/extract hyperparams
    #""" #FIXME: better as a callback (so happens once per iteration)???
    #return sum(dists) #FIXME: ignores saved function
    return -np.sum(scores) #FIXME: ignores saved function


if __name__ == '__main__':

    # get access to data in archive
    from mystic.monitors import Monitor
    m = Monitor()
    m._x,m._y = ds.read_archive(fname)
    xyt = traintest(m._x, m._y, test_size=.2, random_state=42)

    # set bounds
    ub = [.01, 1.0, 1.0]
    lb = [.0001, 0.5, 0.5]

    # configure optimizer
    from mystic.monitors import VerboseLoggingMonitor, Monitor
    mon = VerboseLoggingMonitor(1,1,1)
    ndim = len(ub)
    N = 1 #FIXME: 3 for demo only
    npop = N*ndim
    from mystic.termination import ChangeOverGeneration
    stop = ChangeOverGeneration(1e-4,2) #FIXME: 5 for demo only
    from mystic.solvers import DifferentialEvolutionSolver as Solver
    solver = Solver(ndim,npop)
    #from mystic.solvers import NelderMeadSimplexSolver as Solver
    #solver = Solver(ndim)
    solver.SetRandomInitialPoints(min=lb,max=ub)
    solver.SetStrictRanges(min=lb,max=ub)
    solver.SetGenerationMonitor(mon)
    solver.SetTermination(stop)
    solver.Solve(hypercost, ExtraArgs=xyt, disp=1)

    # get results
    learning, beta1, beta2 = solver.bestSolution
    print('solved: learning_rate = {0}, beta_1 = {1}, beta_2 = {2}'.format(learning, beta1, beta2))

    #""" #NOTE: uncomment if 'dirty' archive block above is uncommented
    # get handles to func_DBs
    if shape[-1]:
        archives = list(map(lambda i: _func_db('est{i}.db'.format(i=i)), range(shape[-1])))
    else:
        archives = _func_db('est.db')

    # check for stored func in func_db, and if not found
    # generate a dummy estimator to store results
    func = read_func(archives)

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
    #"""
