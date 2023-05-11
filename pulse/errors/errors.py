DEFAULT_ERROR = ValueError

def raise_error_if(condition, error=None, msg=None):
    if error is None:
        error = DEFAULT_ERROR
    if condition:
        raise error(msg)

def raise_error_if_not_types(types, **kwargs):
    raise_error_if(not kwargs, msg='kwargs is empty')
    try:
        types = list(types)
    except TypeError:
        types = [types]
    nkwargs = len(kwargs)
    ntypes = len(types)
    if ntypes > nkwargs:
        types = types[:nkwargs]
    elif ntypes < nkwargs:
        raise_error_if(ntypes != 1, msg='Mismatch between number of types and kwargs')
        types = types * nkwargs
    for (k, v), t in zip(kwargs.items(), list(types)):
        if not isinstance(v, t):
            raise TypeError(f'{k}={v} is not of type={t}.')