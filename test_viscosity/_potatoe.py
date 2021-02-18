import potatoe as rb

def _func_db(name):
    """get function db with the name 'name'"""
    if not isinstance(name, (str, (u'').__class__)):
        return getattr(name, '__archive__', name) # need cached == False
    return rb._read_func(name) #, type=type)


def write_func(function, archives):
    """write function to corresponding archives"""
    if getattr(function, '__axis__', None) is None: #XXX: and no len(archives)
        rb._write_func(archives, function, {})
    else:
        [rb._write_func(a, f, {}) for a,f in zip(archives,function.__axis__)]


def read_func(archives): # method?
    """read stored functions from the list of dbs

    Args:
        archives (list[string]): list of names of function archives

    Returns:
        a klepto.archive instance

    Notes:
        The order of the dbs is important, with the index of ``archives`` 
        corresponding to the desired axis. If a db is empty, returns ``None``
        for the empty db. Also, a klepto.archive instance can be provided
        instead of the ``name`` of the db.
    """
    if isinstance(archives, (str, (u'').__class__)):
        archives = _func_db(archives)
    if type(archives).__module__.startswith('klepto.'):
        f = rb.read_func(_func_db(archives))
        if f is None:
            return f
        f = f[0]
        if not hasattr(f, '__axis__'):
            f.__axis__ = None
        return f
    from interpf import interpf_nd
    f = interpf_nd([[1,2],[2,3]],[[1,2],[2,3]],method='thin_plate')
    f.__axis__[:] = [(i if i is None else i[0]) for i in map(rb.read_func, map(_func_db, archives))]
    return f

