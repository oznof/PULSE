import json
from numbers import Number
from typing import Any, Iterable

from ..caster import Caster


class FieldDict(Caster, dict):
    DEFAULT_FIELDS = ('base',)
    DEFAULT_FIELD_DELIMITER = '_' # Field delimiter, after which a subfield is assumed
    # for dumping dict
    DUMPS_WITH_ALL_REFERENCE_KEYS = False  # override any other behavior and display all reference keys
    EXCLUDE_EMPTY_FIELDS = True
    EXCLUDE_EMPTY_VALUES = True


    def __init__(self, fields: Iterable[str] = None, default_field: str = None, delimiter: str = None):
        self._in_init = True
        self._fields = self.DEFAULT_FIELDS if fields is None else fields
        self.raise_error_if(any([not isinstance(field, str) for field in self.fields]))
        self._default_field = self.fields[0] if default_field is None else default_field
        self.raise_error_if(not isinstance(self.default_field, str) and self.default_field not in fields)
        self._delimiter = self.DEFAULT_FIELD_DELIMITER if delimiter is None else delimiter
        self.raise_error_if(not isinstance(self.fd, str))
        # NOTE: the inner dict is just a normal dict, not subject to the same rules as the outer (field) dict
        # so setting or getting from the inner layer has the expected behavior from a normal dict
        dict.__init__(self, ((field, {}) for field in self.fields))
        self._in_init = False

    @property
    def fields(self):
        return self._fields

    @property
    def default_field(self):  # where any other args that don't match fields are stored
        return self._default_field

    @property
    def delimiter(self):
        return self._delimiter

    @property
    def fd(self):
        return self._delimiter

    @property
    def field_delimiter(self):
        return self._delimiter

    @property
    def sep(self):
        return self._delimiter

    @classmethod
    def is_empty(cls, value):
        return (isinstance(value, Iterable) and not any(value)) or not (value or (isinstance(value, bool) or
        isinstance(value, Number)))

    @property
    def exclude_empty_values(self):
        return self.EXCLUDE_EMPTY_VALUES

    @property
    def exclude_empty_fields(self):
        return self.EXCLUDE_EMPTY_FIELDS

    @property
    def excluded_keys(self):
        return ()

    @property
    def excluded_fields(self):
        return ()

    @property
    def excluded_subfields(self):
        return ()

    def excluded_subfields_if(self, key, value):
        return True

    @property
    def reference_dict(self):
        return self

    @property
    def is_self_reference(self):
        return self.reference_dict  == self

    @property
    def dumps_with_all_reference_keys(self):
        return self.DUMPS_WITH_ALL_REFERENCE_KEYS

    @property
    def excluded_keys(self):
        keys = []
        keys += self.to_list_str(self.excluded_keys)
        esf = self.to_list_str(self.excluded_fields)
        for f in esf:
            keys += [k for k, v in self.items() if k.startswith(f + self.fd) and k not in keys]
        if self.exclude_empty_values:
            keys += [k for k, v in self.items() if self.is_empty(v) and k not in keys]
            specials = self.to_list_str(self.excluded_subfields)
            if specials:
                for k, v in self.reference_dict.items():  # iterate over some reference dict, by default itself
                    if k not in keys:  # if not already included and
                        f, s = self.to_field_subfield(k)
                        if s in specials and self.excluded_subfields_if(k, v):
                            keys.append(k)
        return keys

    def to_field_subfield(self, key: str, delimiter: str = None, return_not_in_fields: bool = False):
        if delimiter is None:
            delimiter = self.fd
        field, *subfield = key.split(delimiter, 1)
        not_in_fields = field not in self.fields
        if not_in_fields:
            field, subfield = self.default_field, key
        else:
            subfield = subfield[0]
        return (field, subfield, not_in_fields) if return_not_in_fields else (field, subfield)

    def to_key(self, field: str, subfield: str, delimiter: str = None, keep_default_field=False):
        self.raise_error_if(not isinstance(field, str) or not isinstance(subfield, str))
        if delimiter is None:
            delimiter = self.fd
        field = field if keep_default_field else field.replace(self.default_field, '')
        return delimiter.join(filter(None, (field, subfield)))

    def __setitem__(self, key: str, value: Any):
        if self._in_init:
            dict.__setitem__(self, key, value)
        else:
            self._update(key=key, value=value)

    def __getitem__(self, key: str):
        if key in self.fields:
            return dict.__getitem__(self, key)
        else:
            field, subfield = self.to_field_subfield(key)
            return dict.__getitem__(self, field)[subfield]

    def get(self, key, default=None):
        try:
            return self.__getitem__(key)
        except KeyError:
            return default

    def __delitem__(self, key: str):
        if key in self.fields: # do not delete fields, only reset them
            self[key] = {}
        else:
            f, s = self.to_field_subfield(key)
            del self[f][s]

    def _update(self, key: str = None, field: str = None, subfield: str = None, value: Any = None):
        if isinstance(key, str):
            field, subfield = self.to_field_subfield(key)
        dict.__getitem__(self, field)[subfield] = value
        return self

    def update(self, **kwargs):
        for k, v in kwargs.items():
            self._update(key=k, value=v)

    def keys(self, keep_default_field=False):
        return [self.to_key(field, subfield, keep_default_field=keep_default_field) \
                for field in self.fields for subfield in self[field].keys()]

    def values(self):
        return [v for field in self.fields for v in self[field].values()]

    def items(self, keep_default_field=False):
        for field in self.fields:
            sd = self[field]
            for subfield, v in sd.items():
                k = self.to_key(field, subfield, keep_default_field=keep_default_field)
                yield k, v

    def dump(self):
        return {'data': dict(self), '_fields': self._fields, '_default_field': self._default_field,
                '_delimiter': self._delimiter}
    @classmethod
    def load(cls, d):
        # TODO: put all args in init, if not default fields, but works if correct class.
        new = cls()
        new._in_init = True
        keys_to_exclude = ['_in_init']
        for k, v in d.items():
            if k.startswith('_') and k != '_in_init':  # _in_init should not be in the dict, but keep condition for now
                new.__dict__[k] = v
                keys_to_exclude.append(k)
        d = d.get('data', d)
        for k, v in d.items():
            if k not in keys_to_exclude: # in case 'data' does not exist anymore and it is a flat dict
                new[k] = v
        new._in_init = False
        return new

    def to_dict(self, d=None, excluded_keys=None, **kwargs):
        if d is not None:
            return super().to_dict(d, **kwargs)
        d = dict(self)
        excluded_keys = self.excluded_keys if excluded_keys is None else self.to_list_str(excluded_keys)
        for k in excluded_keys:
            f, s = self.to_field_subfield(k)
            if s in d[f]:
                del d[f][s]
        return d

    def to_reference_dict(self, excluded_keys=None):
        excluded_keys = self.excluded_keys if excluded_keys is None else self.to_list_str(excluded_keys)
        return {k: self[k] for k in self.reference_dict.keys() if k not in excluded_keys}

    @property
    def updated_reference_dict(self):
        return self.to_reference_dict(self.excluded_keys)

    def from_dict(self, d):
        return self.load(d)

    def to_flatten_dict(self, keep_default_field=False):
        return dict(self.items(keep_default_field=keep_default_field))

    def flatd_to_fdict(self, flatd):
        d = {f: {} for f in self.fields}
        for k, v in flatd.items():
            field, subfield = self.to_field_subfield(k)
            d[field][subfield] = v
        return d

    def jdumps(self, **kwargs):
        return json.dumps(self, **kwargs)

    def jloads(self, txt, **kwargs):
        d = json.loads(txt, **kwargs)
        self.from_dict(d)
        return self

    def invalid_value_message(self, field, subfield, value):
        return f'Invalid {value} for {self.to_key(field, subfield, keep_default_field=True)}.'

    def dumps(self, d=None, excluded_keys: Iterable[str] =  None, excluded_fields: Iterable[str] = None,
              exclude_empty_fields: bool = None):
        excluded_keys = self.excluded_keys if excluded_keys is None else self.to_list_str(excluded_keys)
        excluded_keys = () if self.dumps_with_all_reference_keys else excluded_keys
        excluded_fields = self.excluded_fields if excluded_fields is None else self.to_list_str(excluded_fields)
        excluded_fields = () if self.dumps_with_all_reference_keys else excluded_fields
        if not isinstance(d, dict):
            d = self.to_dict(excluded_keys) if self.is_self_reference else \
                self.flatd_to_fdict(self.to_reference_dict(excluded_keys))
                # if other, it is assumed here that reference dict is a "flat" dict
        exclude_empty_fields = self.exclude_empty_fields if exclude_empty_fields is None else exclude_empty_fields
        exclude_empty_fields = False if self.dumps_with_all_reference_keys else exclude_empty_fields
        txt = ''
        for f, subd in d.items():
            if f not in excluded_fields and not (exclude_empty_fields and self.is_empty(subd)):
                txt += f'{f}{self.fd} '
                for s, value in subd.items():
                    if isinstance(value, bool):
                        value = self.bool_to_str(value)
                    txt += f'{s}:{value} '
                txt += ' '
        return txt.rstrip(' ')

    def loads(self, txt, verbose=False):
        fkvps = txt.split('  ')  # split each field, each containing possibly several key value pairs
        if verbose:
            print(fkvps)
            print(self.fields)
        for kvps in fkvps:
            if verbose:
                print("-"*10)
            field, kvps = kvps.split(self.fd, 1)
            for kv in kvps.split(' '):
                if ':' in kv:
                    subfield, value = kv.split(':')
                    if verbose:
                        print(field, subfield, value)
                    self[self.to_key(field, subfield)] = self.try_str_to_bool_or_number(value)
        return self