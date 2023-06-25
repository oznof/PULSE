from numbers import Number
from typing import Callable, Collection, Sized, Iterable, Union
import functools
import operator
import inspect
import itertools
import more_itertools

import numpy as np
import pandas as pd

from ..errors import raise_error_if, raise_error_if_not_types


class Caster:
    @classmethod
    def is_class(cls, obj):
        return inspect.isclass(obj)

    @classmethod
    def is_callable(cls, obj):
        return isinstance(obj, Callable)

    @classmethod
    def is_type(cls, obj):
        return isinstance(obj, type)

    @classmethod
    def is_list_or_tuple(cls, obj):
        return isinstance(obj, list) or isinstance(obj, tuple)

    @classmethod
    def is_singleton(cls, obj):
        return isinstance(obj, str) or not isinstance(obj, Iterable) or (isinstance(obj, Sized) and len(obj) == 1)
               # isinstance(obj, Iterable)
               # not cls.is_list_or_tuple(obj)

    @classmethod
    def all_equal(cls, iter):
        return more_itertools.all_equal(iter)

    @classmethod
    def ichunked(cls, iter, n, strict=False):
        return more_itertools.chunked(iter, n, strict=strict)

    @classmethod
    def ichunked_even(cls, iter, n):
        return more_itertools.chunked_even(iter, n)

    @classmethod
    def idivide(cls, iter, n):
        return more_itertools.divide(n, iter)

    @classmethod
    def isplit_at(cls, iter, cond, maxsplit=-1, keep=False, **kwargs):
        cond = cls.context_wrap('cond', kwargs, default_fn=cond)[0]
        return more_itertools.split_at(iter, cond, maxsplit, keep)

    @classmethod
    def isplit_before(cls, iter, cond, maxsplit=-1, **kwargs):
        cond = cls.context_wrap('cond', kwargs, default_fn=cond)[0]
        return more_itertools.split_at(iter, cond, maxsplit)

    @classmethod
    def isplit_after(cls, iter, cond, maxsplit=-1, **kwargs):
        cond = cls.context_wrap('cond', kwargs, default_fn=cond)[0]
        return more_itertools.split_at(iter, cond, maxsplit)

    @classmethod
    def isplit_into(cls, iter, sizes):
        return more_itertools.split_into(iter, sizes)

    @classmethod
    def isplit_when(cls, iter, cond, maxsplit=-1, **kwargs):
        cond = cls.context_wrap('cond', kwargs, default_fn=cond)[0]
        return more_itertools.split_when(iter, cond, maxsplit)

    @classmethod
    def bucket(cls, iter, key, validator=None, to_dict=True, sort=False, **kwargs):
        bucket = more_itertools.bucket(iter, key, validator)
        if to_dict:
            keys = list(bucket)
            if sort:
                keys = sorted(keys)
            bucket =  cls.to_dict(d=[list(bucket[k]) for k in keys], keys=keys, **kwargs)
        return bucket

    @classmethod
    def minmax(cls, iter, *others, key=None, default=()):
        return more_itertools.minmax(iter, *others, key=key, default=default)

    @classmethod
    def islice(cls, iter, *args): # stop or start, stop [,step]
        return more_itertools.islice_extended(iter, *args)

    @classmethod
    def nth_or_last(cls, iter, n, *args): # n, default
        return more_itertools.nth_or_last(iter, n, *args)

    @classmethod
    def strip(cls, iter, cond, **kwargs):
        cond = cls.context_wrap('cond', kwargs, default_fn=cond)[0]
        return more_itertools.strip(iter, cond)

    @classmethod
    def lstrip(cls, iter, cond, **kwargs):
        cond = cls.context_wrap('cond', kwargs, default_fn=cond)[0]
        return more_itertools.lstrip(iter, cond)

    @classmethod
    def rstrip(cls, iter, cond, **kwargs):
        cond = cls.context_wrap('cond', kwargs, default_fn=cond)[0]
        return more_itertools.rstrip(iter, cond)

    @classmethod
    def filter(cls, iter, cond, **kwargs):
        cond = cls.context_wrap('cond', kwargs, default_fn=cond)[0]
        return filter(cond, iter)

    @classmethod
    def filterfalse(cls, iter, cond, **kwargs):
        cond = cls.context_wrap('cond', kwargs, default_fn=cond)[0]
        return itertools.filterfalse(cond, iter)

    @classmethod
    def filter_except(cls, iter, cond, *exceptions, **kwargs):
        cond = cls.context_wrap('cond', kwargs, default_fn=cond)[0]
        return more_itertools.filter_except(cond, iter, *exceptions)

    @classmethod
    def map(cls, iter, fn, **kwargs):
        fn = cls.context_wrap('fn', kwargs, default_fn=fn)[0]
        return map(fn, iter)

    @classmethod
    def map_except(cls, iter, fn, *exceptions, **kwargs):
        fn = cls.context_wrap('fn', kwargs, default_fn=fn)[0]
        return more_itertools.map_except(fn, iter, *exceptions)

    @classmethod
    def map_if(cls, iter, cond, fn, **kwargs):
        cond, cond_kwargs, kwargs = cls.context_wrap('cond', kwargs, default_fn=cond)
        fn, fn_kwargs, kwargs = cls.context_wrap('fn', kwargs, default_fn=fn)
        other_fn, other_fn_kwargs, kwargs = cls.context_wrap('other_fn', kwargs, default_fn=lambda x: x)
        return more_itertools.map_if(iter, cond, fn, other_fn)

    @classmethod
    def map_reduce(cls, iter, fn=None, kw=None, v=None, kw_only=False, v_only=False, **kwargs):
        kw_fn, kw_kwargs, kwargs = cls.context_wrap('kw', kwargs, default_fn=kw or (lambda z: z))
        if kw_only:
            _kw_fn = kw_fn
            kw_fn = lambda k: _kw_fn(k[0])
        v_fn, v_kwargs, kwargs = cls.context_wrap('v', kwargs, default_fn=v or (lambda z: z))
        if v_only:
            _v_fn = v_fn
            v_fn = lambda a: _v_fn(a[1])
        if fn is not None: # if it is none there is no need to wrap and summarization is skipped altogether by using None
            fn, fn_kwargs, kwargs = cls.context_wrap('fn', kwargs, default_fn=fn)
        return more_itertools.map_reduce(iter, kw_fn, v_fn, fn)

    @classmethod
    def unzip(cls, iter):
        return more_itertools.unzip(iter)

    @classmethod
    def partition(cls, iter, cond, to_list=True, **kwargs):
        # first false, second true
        cond, cond_kwargs, kwargs = cls.context_wrap('cond', kwargs, default_fn=cond)
        partition = more_itertools.partition(cond, iter)
        #group_kwargs = {k.lstrip('group_'): v for k, v in kwargs.items() if k.startswith('group_')}
        #[kwargs.pop('group_' + k) for k in group_kwargs]
        if to_list:
            group_kwargs, kwargs = cls.filter_kwargs('group', kwargs)
            partition = cls.to_list([cls.to_list(p, **kwargs) for p in partition], **group_kwargs)
        return partition

    @classmethod
    def windowed(cls, iter, n, fillvalue=None, step=1, to_list=True, **kwargs):
        windows = more_itertools.windowed(iter, n, fillvalue, step)
        if to_list:
            windows = cls.to_list([cls.to_list(w, **kwargs) for w in windows])
        return windows

    @classmethod
    def stagger(cls, iter, offsets=(-1, 0, 1), longest=False, fillvalue=None, to_list=True, **kwargs):
        s = more_itertools.stagger(iter, offsets, longest, fillvalue)
        if to_list:
            s = cls.to_list(s, **kwargs)
        return s

    @classmethod
    def ipairwise(cls, iter, to_list=False, **kwargs):
        pairs = more_itertools.pairwise(iter)
        if to_list:
            pairs = cls.to_list(pairs, **kwargs)
        return pairs

    @classmethod
    def padded(cls, iter, fillvalue=None, size=None, next_multiple=False, to_list=True, **kwargs):
        seq = more_itertools.padded(iter, fillvalue, size, next_multiple)
        if to_list:
            seq = cls.to_list(seq, **kwargs)
        return seq

    @classmethod
    def adjacent(cls, iter, cond, distance=1, to_list=True, remove_false=True, **kwargs):
        cond, cond_kwargs, kwargs = cls.context_wrap('cond', kwargs, default_fn=cond)
        bool_item_pairs = more_itertools.adjacent(cond, iter, distance)
        if to_list:
            bool_item_pairs = cls.to_list([cls.to_list(pair, **kwargs) for pair in  bool_item_pairs])
            if remove_false:
                return cls.to_list(bool_item_pairs, filter=lambda x: x[0], map=lambda x: x[1])
        return bool_item_pairs

    @classmethod
    def before_and_after(cls, iter, cond, **kwargs):
        _type = type(iter)
        cond = cls.context_wrap('cond', kwargs, default_fn=cond)[0]
        before, after = more_itertools.before_and_after(cond, iter)
        if _type is list:
            return cls.to_list(before, **kwargs), cls.to_list(after, **kwargs)
        if _type is str:
            return ''.join(before), ''.join(after)
        return before, after

    @classmethod
    def locate(cls, iter, cond, window_size=None, to_list=True, return_value=False, **kwargs):
        cond, cond_kwargs, kwargs = cls.context_wrap('cond', kwargs, default_fn=cond)
        idx = more_itertools.locate(iter, cond, window_size)
        if to_list:
            idx = cls.to_list(idx, **kwargs)
        if return_value:
            value_kwargs, kwargs = cls.filter_kwargs('value', kwargs)
            value = operator.itemgetter(*idx)(iter)
            return cls.to_list(value, **value_kwargs) if to_list else value
        return idx

    @classmethod
    def replace(cls, iter, cond, subs, count=None, window_size=1, to_list=True, **kwargs):
        cond, cond_kwargs, kwargs = cls.context_wrap('cond', kwargs, default_fn=cond)
        subs = cls.to_list(subs)
        iter = more_itertools.replace(iter, cond, subs, count, window_size)
        if to_list:
            iter = cls.to_list(iter, **kwargs)
        return iter

    @classmethod
    def repeat_each(cls, iter, n, to_list=True, **kwargs):
        out =  more_itertools.repeat_each(iter, n)
        if to_list:
            out = cls.to_list(out, **kwargs)
        return out

    @classmethod
    def repeat_last(cls, iter, n, to_list=True, **kwargs):
        out =  more_itertools.repeat_last(iter, n)
        if to_list:
            out = cls.to_list(out, **kwargs)
        return out

    @classmethod
    def collapse(cls, iter, base_type=None, levels=None, **kwargs):
        _type = type(iter)
        out = more_itertools.collapse(iter, base_type, levels)
        if isinstance(_type, list):
            return cls.to_list(out, **kwargs)
        return out

    @classmethod
    def prepend(cls, iter, value, **kwargs):
        _type = type(iter)
        if _type is str:
            if isinstance(value, bool):
                value = cls.bool_to_str(value)
            else:
                value = cls.module_obj_to_str(value)
        out = more_itertools.prepend(value, iter)
        if _type is list:
            return cls.to_list(out, **kwargs)
        if _type is str:
            return ''.join(out)
        return out

    @classmethod
    def exactly_n(cls, iter, cond, n, **kwargs):
        cond = cls.context_wrap('cond', kwargs, default_fn=cond)[0]
        return more_itertools.exactly_n(iter, n, cond)

    @classmethod
    def all_unique(cls, iter, key=None):
        return more_itertools.all_unique(iter, key)


    @classmethod
    def list_unique(cls, seq):
        if cls.is_singleton(seq):
            return seq
        return type(seq)(dict.fromkeys(seq))

    @classmethod
    def list_remove(cls, seq, items):
        is_tuple = isinstance(seq, tuple)
        seq = cls.to_list(seq)
        for item in cls.to_list(items):
            while item in seq:
                seq.remove(item)
        return tuple(seq) if is_tuple else seq

    @classmethod
    def only_seq(cls, seq, method, filter=True, on='kw', cast=True, **kwargs):
        seq_type = type(seq)
        method = getattr(seq_type, method) if isinstance(method, str) else method
        filter_mode = filter
        filter = cls.filter if filter_mode else cls.filterfalse
        if seq_type is list or seq_type is tuple:
            if cast:
                seq = cls.to_list_str(seq)
            return seq_type(filter(seq, method))
        elif isinstance(seq, dict):
            filter_name = '' if filter_mode else 'false'
            if on in ['kw', 'v']:
                f = (lambda x: method(str(x))) if cast else method
                kwargs.update({f'filter{filter_name}_{on}': f})
            elif on == 'both':
                f = ((lambda x: method(str(x[0])) and method(str(x[1]))) if cast
                     else (lambda x: method(x[0]) and method(x[1])))
                kwargs.update({f'filter{filter_name}_{on}': f})
            return cls.to_dict(seq, **kwargs)
        return seq_type().join(filter(str(seq) if cast else seq, method))

    @classmethod
    def only_digits(cls, seq, **kwargs):
        return cls.only_seq(seq, method=str.isdigit, **kwargs)

    @classmethod
    def only_alpha(cls, seq, **kwargs):
        return cls.only_seq(seq, method=str.isalpha, **kwargs)

    @classmethod
    def only_upper(cls, seq, **kwargs):
        return cls.only_seq(seq, method=str.isupper, **kwargs)

    @classmethod
    def only_lower(cls, seq, **kwargs):
        return cls.only_seq(seq, method=str.islower, **kwargs)

    @classmethod
    def only_prefix(cls, seq, prefix, delimiter=None, **kwargs):
        seq = cls.only_seq(seq, method=(lambda x: x.startswith(prefix)), **kwargs)
        if delimiter is not None:
            seq = cls.strip_prefix(seq, prefix, delimiter)
        return seq

    @classmethod
    def strip_prefix(cls, seq, prefix, delimiter=None):
        if delimiter is None:
            delimiter = ''
        fn = lambda s: s.lstrip(prefix + delimiter)
        seq = fn(seq) if isinstance(seq, str) else type(seq)(map(fn, seq))
        return seq

    @classmethod
    def to_dict(cls, d, *args, kw=None, **kwargs): # lambda_args=lambda_args, lambda_kwargs=lambda_kwargs
        # print("IN DICT WITH: ", d, args, kw, kwargs)
        _type = kwargs.get('dict', dict)
        _kwargs, kwargs = cls.filter_kwargs('dict', kwargs)
        __dict = lambda __d :  _type(__d, **_kwargs)
        _dict = lambda dd: (dd if isinstance(dd, _type) else __dict(dd))
        if cls.is_list_or_tuple(d):
            d = cls.to_list(d)
            kw = kwargs.pop('keys', kw)
            kw = range(len(d)) if kw is None else kw
            d = zip(kw, d)
        d = _dict(d)
        if kwargs:
            functions = {'filter': cls.filter, 'filterfalse': cls.filterfalse, 'map': cls.map,
                         'mapif': cls.map_if}
            # NOTE: make sure signatures of lambdas are different
            filter_wrappers = {'': lambda _fn, x, f=1: _fn(x), 'kw': lambda _fn, y, f=1: _fn(y[0]),
                               'v': lambda _fn, z, f=1: _fn(z[1])}
            wrappers = {'': lambda _fn, x, f=2: _fn(x), 'kw': lambda _fn, y, f=2: (_fn(y[0]), y[1]),
                        'v': lambda _fn, z, f=2: (z[0], _fn(z[1])) }
            words = ['filter', 'filter_kw', 'filter_v', 'filterfalse', 'filterfalse_kw', 'filterfalse_v',
                     'map', 'map_kw', 'map_v', 'mapif', 'mapif_kw', 'mapif_v'] # todo: check mapif
            words = [kw for kw in kwargs if kw in words]
            for word in words:
                function_word, *selector_word = word.rsplit('_', 1)
                selector_word = selector_word[0] if selector_word else ''
                dwrap = filter_wrappers if function_word.startswith('filter') else wrappers
                selector = dwrap.get(selector_word, lambda u: u)
                function = functions[function_word]
                if function_word.startswith('filter') or not function_word.endswith('if'):
                    fn, word_kwargs, kwargs = cls.context_wrap(word, kwargs)
                    d = _dict(function(d.items(), lambda s: selector(fn, s)))  # .items())
                else:
                    # mapif behaves differently, mapif checks keys and applies on values
                    _cond = kwargs.pop(word)
                    word_kwargs, kwargs = cls.filter_kwargs(word, kwargs)
                    _cond, word_cond_kwargs, word_kwargs = cls.context_wrap('cond', word_kwargs,
                                                                            default_fn=_cond)
                    _fn, word_fn_kwargs, word_kwargs = cls.context_wrap('fn', word_kwargs)
                    _other_fn, word_other_fn_kwargs, word_kwargs = cls.context_wrap('other_fn', word_kwargs,
                                                                                    default_fn = lambda ot: ot)
                    if word == 'mapif' and word_kwargs.pop('vk', False):
                        d = _dict(function(d.items(), cond=lambda a: _cond(a[1]), fn=lambda b: (_fn(b[0]), b[1]),
                                           other_fn=lambda c: (_other_fn(c[0]), c[1])))
                    elif word == 'mapif' and word_kwargs.pop('kv', True):

                        d = _dict(function(d.items(), cond=lambda a: _cond(a[0]), fn=lambda b: (b[0], _fn(b[1])),
                                           other_fn=lambda c: (c[0], _other_fn(c[1]))))
                    else:
                        cond_selector = filter_wrappers[selector_word]
                        d = _dict(function(d.items(), cond=lambda a: cond_selector(_cond, a),
                                           fn=lambda b: selector(_fn, b),
                                           other_fn=lambda c: selector(_other_fn, c)))
            kwargs.setdefault('reduce_wrap', False)
            fn, red_kwargs, kwargs = cls.context_wrap('reduce', kwargs)
            if red_kwargs or fn:
                red_kwargs.setdefault('kw_only', True)
                red_kwargs.setdefault('v_only', True)
                d = __dict(cls.map_reduce(d.items(), fn=fn, **red_kwargs)) # make sure it is not defaultdict
            cls.raise_error_if(kwargs, error=RuntimeError, msg='Not all kwargs were consumed.\n'
                                                               f'Remaining: {kwargs}')
        return d

    @classmethod
    def to_list(cls, seq, *args, **kwargs):
        # print("IN LIST WITH: ", seq, args, kwargs)
        _type = kwargs.get('list', list)
        _kwargs, kwargs = cls.filter_kwargs('list', kwargs)
        _list = lambda x: ([x] if (_type is list and not isinstance(x, list) and cls.is_singleton(x))
                           else (x if isinstance(x, _type) else _type(x, **_kwargs)))
        seq = _list(seq)
        for arg in args:
            seq += _list(arg)
        if kwargs:
            # NOTE: for more flexibility, order of operations is determined based on the call
            functions = {'filter': cls.filter, 'filterfalse': cls.filterfalse, 'map': cls.map,
                         'mapif': cls.map_if}
            words = [kw for kw in kwargs if kw in functions]
            for word in words:
                if word.startswith('filter') or not word.endswith('if'):
                    fn, word_kwargs, kwargs = cls.context_wrap(word, kwargs)
                    seq = _list(functions[word](seq, fn))
                else:
                    kwargs.setdefault(f'{word}_wrap', False)
                    cond, word_kwargs, kwargs = cls.context_wrap(word, kwargs)
                    # pass in kwargs, still needs to be wrapped
                    seq = _list(functions[word](seq, cond=cond, **word_kwargs))
            kwargs.setdefault('reduce_wrap', False)
            fn, red_kwargs, kwargs = cls.context_wrap('reduce', kwargs)
            if fn:
                has_kw = 'kw' in red_kwargs
                if not has_kw:
                    red_kwargs['kw'] = lambda x: 0
                seq = cls.map_reduce(seq, fn=fn, **red_kwargs)
                seq = dict(seq) if has_kw else _list(seq[0])
        return seq

    @classmethod
    def to_item_if_singleton(cls, seq):
        return seq[0] if cls.is_list_or_tuple(seq) and len(seq) == 1 else seq

    @classmethod
    def to_sized_list(cls, obj, size, fill='append', other=None, **kwargs):
        cls.raise_error_if(fill not in ['append', 'prepend'])
        obj = cls.to_list(obj)[:size]
        ld = size - len(obj)
        if ld > 0:
            if other is not None:
                if isinstance(other, Collection) and not isinstance(other, str):
                    other = other[:size]
                    raise_error_if(len(other) < size)
                else:
                    other = [other, ] * size
            if fill == 'append':
                obj += [obj[-1], ] * ld if other is None else other[-ld:]
            else:
                obj = ([obj[0], ] * ld if other is None else other[:ld]) + obj
        return obj

    @classmethod
    def to_list_str(cls, obj):
        return [str(obj)] if isinstance(obj, str) else (cls.to_type(str, *obj) if obj else [])

    @classmethod
    def gen_flat_items(cls, d, field, sep, depth=1, max_depth=None):
        for k, v in d.items():
            key = field + sep + k if field else k
            if isinstance(v, dict) and (max_depth is None or depth <= max_depth):
                yield from cls.to_flat_dict(v, key, sep, depth + 1, max_depth).items()
            else:
                yield key, v

    @classmethod
    def to_flat_dict(cls, d: dict, field='', sep='::', depth=1, max_depth=None):
        try:
            sep = getattr(d, 'sep')
            d = d.to_flatten_dict()
            depth = 1
        except (AttributeError, KeyError):
            pass
        return dict(cls.gen_flat_items(d, field, sep, depth, max_depth))

    @classmethod
    def to_type(cls, type, *args):
        return [type(arg) for arg in args]
    
    @classmethod
    def to_types(cls, types, *args, **kwargs):
        return [t(a) for t, a in zip(cls.to_sized_list(types, len(args), **kwargs), args)]

    @classmethod
    def bool_to_str(cls, value):
        return str(value).lower()

    @classmethod
    def str_to_bool(cls, txt):
        return txt.lower() == 'true'

    @classmethod
    def is_bool_or_str_true(cls, obj):
        return (isinstance(obj, bool) and obj) or (isinstance(obj, str) and obj.lower() == 'true')

    @classmethod
    def is_bool_or_str_false(cls, obj):
        return (isinstance(obj, bool) and not obj) or (isinstance(obj, str) and obj.lower() == 'false')

    @classmethod
    def str_to_number(cls, txt):
        for i, fn in enumerate([int, float]):
            try:
                return fn(txt)
            except ValueError:
                if i == 1:  # raise error if both fail
                    raise ValueError

    @classmethod
    def try_str_to_bool_or_number(cls, txt, error_on_str=False):
        if txt.lower() == 'false':
            return False
        if txt.lower() == 'true':
            return True
        try:
            for i, fn in enumerate([int, float]):
                try:
                    return fn(txt)
                except ValueError:
                    if i == 1:  # raise error if both fail
                        raise ValueError
        except ValueError:
            cls.raise_error_if(error_on_str, msg='Not bool nor number.')
            return txt

    @classmethod
    def is_bool_or_number(cls, value):
        if isinstance(value, bool) or isinstance(value, Number):
            return True
        try:
            cls.try_str_to_bool_or_number(value, error_on_str=True)
            return True
        except ValueError:
            return False

    @classmethod
    def value_from_str(cls, txt, prefix, type=str, delimiter=','):
        txt, prefix, delimiter = cls.to_type(str, txt, prefix, delimiter)
        # value = type(txt.split(prefix)[1].split(delimiter)[0])
        for t in txt.split(delimiter):
            if t.startswith(prefix):
                value = type(t.replace(prefix, ''))
                return value
        raise ValueError(f'Prefix {prefix} not in txt={txt}')

    @classmethod
    def replace_value_in_str(cls, txt, value, prefix, delimiter=','):
        txt, prefix, delimiter = cls.to_type(str, txt, prefix, delimiter)
        # b, a = txt.split(prefix, 1)
        # a = a.split(delimiter, 1)[-1] if delimiter in a else ''
        # return f"{b}{prefix}{value}{a}"
        return delimiter.join([(f'{prefix}{value}' if t.startswith(prefix) else t) for t in txt.split(delimiter)])

    @classmethod
    def to_parsed(cls, seq, field=None, prefixes=None, types=None, delimiter=',',
                  filters=None, fill='append', unique=False, types_auto=False):
        parsed = {}
        seq = cls.to_list_str(seq)
        if field:
            seq = cls.only_prefix(seq, field, delimiter=delimiter)
        if seq:
            singleton = cls.is_singleton(seq)
            len_subf = len(seq[0].split(delimiter))
            if not types:
                types = None if types_auto else str
            types = cls.to_sized_list(types, len_subf, fill=fill)
            if filters is None:
                filters = lambda x: cls.only_seq(x, lambda y: not y.isupper())
            filters = cls.to_sized_list(filters, len_subf, fill=fill)
            if prefixes:
                prefixes = cls.to_list_str(prefixes)[:len_subf]
                fil = [lambda x, p=p: cls.strip_prefix(x, p) for p in prefixes]
                filters = cls.to_sized_list(fil, len_subf, fill=fill, other=filters)
            for s in seq:
                for sub, f, t in zip(s.split(delimiter), filters, types):
                    v = f(sub)
                    p = sub.rstrip(v)
                    if p not in parsed:
                        parsed[p] = [t(v) if t else cls.try_str_to_bool_or_number(v)]
                    else:
                        parsed[p].append(t(v))
            if singleton:
                parsed = {k: v[0] for k, v in parsed.items()}
            elif unique:
                parsed = {k: cls.list_unique(v) for k, v in parsed.items()}
        return parsed

    @classmethod
    def parsed_to_str(cls, parsed, field=None, order=None, delimiter=',',
                      named_value_to_str: Union[dict, Callable] = None,
                      value_delimiter=None, sort=False, ascending=True,
                      remove_prefixes=None,
                      remove_value_if_true=False,
                      remove_prefix_if_str=False, remove_prefix_if_false=False,
                      prefix_to_str: Union[dict, Callable] = None):
        # print("PARSED TO STR: ", value_to_str)
        prefixes = cls.to_list_str(parsed.keys())
        remove_prefixes = cls.to_list_str(remove_prefixes) if remove_prefixes else []
        len_subf = len(prefixes)
        default_order = list(range(len_subf))
        if order is None:
            order = default_order
        elif order == 'reverse':
            order = default_order[::-1]
        else:
            order = cls.list_unique(list(order))[:len_subf]
            order += list(set(default_order) - set(order))
        if named_value_to_str is None:
            named_v_to_str = cls.named_value_to_str
        elif isinstance(named_value_to_str, dict):
            # try dict first before going with class default
            if field in named_value_to_str:
                named_value_to_str = named_value_to_str[field]
                vp = ''
            else:
                vp = '' if field is None or value_delimiter is None else field + value_delimiter
            named_v_to_str = (lambda k, v: named_value_to_str[vp+k].get(v, cls.named_value_to_str(k, v))
                              if (vp+k) in named_value_to_str
                              else named_value_to_str.get(v, cls.named_value_to_str(k, v)))
        else:
            named_v_to_str = named_value_to_str
        prefixes = [prefixes[o] for o in order]
        parsed = {prefix: cls.to_list(parsed[prefix]) for prefix in prefixes}
        pprefix_to_str = ((lambda x: x) if prefix_to_str is None else
            (lambda x: prefix_to_str.get(x, x)) if isinstance(prefix_to_str, dict) else prefix_to_str)
        p_to_str = lambda p, v : ('' if (p in remove_prefixes or
                                         not named_v_to_str(p, v) or
                                         (remove_prefix_if_str and not cls.is_bool_or_number(v)) or
                                         (remove_prefix_if_false and cls.is_bool_or_str_false(v)))
                                  else pprefix_to_str(p))
        seq = [delimiter.join(filter(None, [p_to_str(p, v) +
                               ('' if remove_value_if_true and cls.is_bool_or_str_true(v) else named_v_to_str(p, v))
                               for p, v in zip(prefixes, values)]))
               for values in zip(*parsed.values())]
        if sort:
            seq = sorted(seq, reverse=not ascending)
        if field:
            seq = [field + delimiter + s for s in seq]
        seq = seq[0] if cls.is_singleton(seq) else seq
        return seq

    @classmethod
    def module_obj_to_str(cls, obj):
        name = str(obj)
        if name.count("'") > 1:
            name = name.split("'", 2)[1]
        return name

    @classmethod
    def str_to_module_obj(cls, txt):
        if not isinstance(txt, str):
            return txt
        if '.' not in txt:
            return globals()[txt]
        else:
            if "'" in txt:
                txt = txt.split("'", 2)[1]
            m, o = txt.rsplit('.', 1)
            module = __import__(m, fromlist=[o])
            return getattr(module, o)

    @classmethod
    def named_value_to_str(cls, key, value):
        return cls.bool_to_str(value) if isinstance(value, bool) else cls.module_obj_to_str(value)

    @classmethod
    def value_to_str(cls, value):
        return cls.named_value_to_str(key=None, value=value)

    @classmethod
    def df_unique_level_values(cls, df, level: Union[int, str] = 0, axis=1, to_list=True):
        index = df.index if axis == 0 else df.columns
        values = index.get_level_values(level).unique()
        if to_list:
            values = values.tolist()
        return values

    @classmethod
    def to_stats_df(cls, res):
        columns = pd.MultiIndex.from_product((res.keys(), ('mean', 'std', 'median', 'q1', 'q3', 'iqr')))
        new = pd.DataFrame(columns=columns)
        for k, v in res.items():
            v = np.array(v)
            if v.shape[-1] < len(new.index):
                v = np.hstack((np.full((v.shape[0], len(new.index) - v.shape[-1]), np.nan), v))
            new[(k, 'mean')] = pd.Series(np.mean(v, axis=0))
            new[(k, 'std')] = pd.Series(np.std(v, axis=0))
            new[(k, 'median')] = pd.Series(np.median(v, axis=0))
            q1 = np.quantile(v, 0.25, axis=0)
            new[(k, 'q1')] = pd.Series(q1)
            q3 = np.quantile(v, 0.75, axis=0)
            new[(k, 'q3')] = pd.Series(q3)
            new[(k, 'iqr')] = pd.Series(q3 - q1)
        return new

    @classmethod
    def raise_error_if(cls, condition, error=None, msg=None):
        return raise_error_if(condition, error=error, msg=msg)

    @classmethod
    def raise_error_if_not_types(cls, types, **kwargs):
        return raise_error_if_not_types(types, **kwargs)

    @classmethod
    def get_operator(cls, op=None, type_check=False, reduce=False):
        if cls.is_callable(op):
            return op
        if type_check:
            return any if reduce else isinstance
        if op is None:
            return (lambda x: x) if reduce else operator.eq
        if isinstance(op, str):
            try:
                op = getattr(cls, op)
            except (AttributeError, KeyError):
                try:
                    op = getattr(operator, op)
                except AttributeError:
                    _op = op
                    op = lambda x, y: getattr(x, _op)(y)
        return op

    @classmethod
    def wrap(cls, obj, with_args=False, type_check=False, conditional=False,
             return_condition=False, **kwargs):
        type_check = type_check or all(list(map(lambda x: cls.is_type(x) or x is None,
                                                cls.to_list(obj))))
        if not cls.is_callable(obj) or type_check or 'operator' in kwargs:
            type_checker = lambda _x: _x if not type_check or cls.is_type(_x) else type(_x)
            if cls.is_singleton(obj) and not cls.is_list_or_tuple(obj):
                _this = type_checker(obj)
                op = cls.get_operator(kwargs.pop('operator', None), type_check=type_check)
                obj = lambda x, *a, **b: op(x, _this, *a, **b)
                #obj = (lambda x, *a, **b: isinstance(x, _this)) if type_check else\
                #      (lambda x, *a, **b: op(x, _this, *a,  **b))
            else:
                _this = list(map(type_checker, list(obj)))
                type_check = type_check
                _reduce = cls.get_operator(kwargs.pop('reduce', None),
                                           type_check=type_check or not all(list(map(cls.is_callable, _this))),
                                           reduce=True)
                op = kwargs.pop('operator', None)
                _ops = list(map(lambda p: cls.get_operator(p, type_check=type_check), cls.to_sized_list(op, len(_this))))
                obj = (lambda x: _reduce([isinstance(x, t) for t in _this])) if type_check else \
                      (lambda x, *a, **b: (_reduce([t(x, *a, **b) if cls.is_callable(t) else o(x, t, *a, **b)
                                          for t, o in zip(_this, _ops)])))
            if type_check:
                with_args, kwargs = False, {}
        elif conditional:
            return (obj, False) if return_condition else obj
        _obj = obj
        fn = (lambda *args, **kws: _obj(*args, **kwargs)) if with_args else\
             (lambda *args, **kws: _obj(args[0], **kwargs))
        return (fn, True) if return_condition else fn

    @classmethod
    def context_wrap(cls, context, kwargs, default_fn=None, **other):
        context_fn = kwargs.get(context, default_fn)
        context_kwargs, kwargs = cls.filter_kwargs(context, kwargs, **other)
        context_wrap = context_kwargs.pop('wrap', kwargs.get('wrap', True))
        if context_wrap:
            context_fn = cls.wrap(context_fn, **context_kwargs)
        return context_fn, context_kwargs, kwargs

    @classmethod
    def filter_kwargs(cls, name, kwargs, delimiter='_', remove=True):
        prefix = name + delimiter
        new_kwargs = {k.replace(prefix, ''): v for k, v in kwargs.items() if k.startswith(prefix)}
        if remove:
            [kwargs.pop(prefix + k) for k in new_kwargs]
            if name in kwargs:
                kwargs.pop(name)
        return new_kwargs, kwargs if remove else new_kwargs

    @classmethod
    def partial(cls, *args, **kwargs):
        return functools.partial(*args, **kwargs)