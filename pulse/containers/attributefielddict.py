from typing import Any, Iterable

from .fielddict import FieldDict


class AttributeFieldDict(FieldDict):
    """Extending FieldDict with an interface of items via attributes"""
    DEFAULT_ATTR_DELIMITER = '_'

    def __init__(self, fields: Iterable[str] = None, default_field: str = None,
                 field_delimiter: str = None, attr_delimiter: str = None):
        object.__setattr__(self, '_in_init', True)
        self._cls_attrs = self.get_cls_attrs()
        FieldDict.__init__(self, fields=fields, default_field=default_field, delimiter=field_delimiter)
        self._attr_delimiter = self.default_attr_delimiter if attr_delimiter is None else attr_delimiter
        self.raise_error_if(not isinstance(self.ad, str))
        self._property_setters =  self.get_property_setters()
        self._property_setter_names = set(self.property_setters.keys())
        self._in_init = False

    @classmethod
    def get_cls_attrs(cls):
        d = {}
        for c in reversed(cls.__mro__):
            d.update({k: v for k, v in c.__dict__.items() if not k.startswith('__')})
        return d

    @property
    def cls_attrs(self):
        return self._cls_attrs

    @property
    def attr_delimiter(self):
        return self._attr_delimiter

    @property
    def ad(self):
        return self._attr_delimiter

    @property
    def property_setters(self):
        return self._property_setters

    @property
    def property_setter_names(self):
        return self._property_setter_names

    @classmethod
    def get_property_setters(cls):
        setters = {}
        for c in reversed(cls.__mro__):
            setters.update({attr: value.fset for attr, value in vars(c).items()
                            if isinstance(value, property) and value.fset is not None})
        return setters

    def is_writable_attr(self, name):
        return name in self.property_setter_names

    # NOTE: default names are defined based on reference dict (args) that led to settings
    @property
    def name(self):
        return self.custom_name if self.has_custom_name else self.default_name

    @property
    def custom_name(self):
        return self.reference_dict['name']

    @property
    def has_custom_name(self):
        return isinstance(self.reference_dict['name'], str)

    @property
    def updated_args(self):
        return self.to_reference_dict(self.excluded_keys)

    @property
    def default_name(self):  # can only be generated after loading
        self.raise_error_if(not self.has_loaded, error=AttributeError)
        name = self.reference_dict.get('name')
        if not isinstance(name, str):
            name = self.dumps()
        return name

    # NOTE: item and attribute should behave similarly, calling the custom setter if it exists
    # and storing in the inner dicts. to some extent the same applies to getters  (see below)
    # that is, make getattr less case sensitive and link it to getitem
    # the convention here is to use upper for class attributes and lower for instance attributes

    def __getattr__(self, name):
        try:
            return object.__getattribute__(self, name) # normal
        except AttributeError:
            cls_attrs = self._cls_attrs
            for cls_name in [name.upper()]:  # 2. name upper in class
                try:
                    return cls_attrs[cls_name]
                except KeyError:
                    pass
            return self[name]  # 3: getitem with given name
        # cls_name = f'_{self.__class__.__name__}'
        # if name.startswith(cls_name):  # this handles dunder attributes
        #     name = name.replace(cls_name, '')
        #     return self.__dict__[name]

    # IMPORTANT: if attribute setters exist (custom), these must set values in inner dicts and not the outer
    # field, subfield = self.attr_to_field_subfield(name)
    # self[field][subfield] = value
    def __setattr__(self, name: str, value: Any):
        if name.startswith('_') or self._in_init or name.upper() == name:  # handle private/uppercase attributes normally
            self.__dict__[name] = value
        else:
            # property name and key conversion: by default this is the same, but is required when
            # attr delimiter and field delimiter are not the same, e.g. using '_' for attributes
            # and using '.' for field delimiter, that is keys in args
            field, subfield = self.attr_to_field_subfield(name)
            self._update(field=field, subfield=subfield, value=value)

    def __delattr__(self, name):
        # TODO: if this is not wanted, override it.
        # cannot delete "normal" attributes, only items in the dict
        # NOTE: by construction (see getattr), items in the dict are given an "attribute view".
        self.__delitem__(name)

    def to_attr_name(self, key):
        return self.to_key(*self.to_field_subfield(key), delimiter=self.ad)

    def attr_to_field_subfield(self, name):
        return self.to_field_subfield(name, delimiter=self.ad)

    def attr_to_key(self, name):
        return self.to_field_subfield(name, delimiter=self.ad)

    def field_subfield_to_attr_name(self, field, subfield):
        return self.to_key(field=field, subfield=subfield, delimiter=self.ad)

    def _update(self, key: str = None, field: str = None, subfield: str = None, value: Any = None):
        # NOTE: values are put first in the (inner) dict, and only then the setters are called,
        # so that if one does not want to modify the values further, there is no need to define a custom setter
        # it only makes sense to handle further processing of values as well as conditional setting of other
        # items
        # the name of the property should match key up to the respective delimiter (by default they are the same, '_')
        # IMPORTANT: BE CAREFUL NOT TO CALL THE SAME SET ITEM IN A PROPERTY SETTER,
        # use self[field][subfield] = ...,
        # and NOT self[key] = ... as this would call _update again
        # if a key belongs to the default field, the attribute name should not have the field in its name
        # e.g., default_field = base, then for subfield = path, name the property as path and not base_path,
        # otherwise, the field should be included, separated by the corresponding attr delimiter
        # print(key, field, subfield, value)
        if key is None:
            key = self.to_key(field=field, subfield=subfield)
        field, subfield = self.to_field_subfield(key)
        self[field][subfield] = value
        name = self.field_subfield_to_attr_name(field, subfield)
        if name in self.property_setter_names:
            self.property_setters[name](self, value)
        return self

    def dump(self):
        d = super().dump()
        d.update({'_attr_delimiter': self._attr_delimiter})
        return d