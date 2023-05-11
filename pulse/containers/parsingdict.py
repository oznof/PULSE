import argparse
from typing import Any, Callable, Dict, List, Iterable, Tuple, Union 

from .attributefielddict import AttributeFieldDict

INVALID_PREFIX = '*<>?#'


class ParsingDict(AttributeFieldDict):

    IMMUTABLE_AFTER_LOADING = True
    # 1.
    # initially, this class parses normal commands via argparse (parse method below), default commands should be
    # given by overriding the property default_commands and conditional commands given in get_commands_extra
    # there can be as many extra commands as wanted as long as get_commands_extra returns a next commands getter

    # 2.
    DEFAULT_SUBFIELD_DELIMITER = ','  # delimiter for parsing strings in given special subfields
    # it assumes a parse dict name is defined as f'{field}{attr_delimiter}{subfield}'_dict
    # either all upper or all lower: see getattr
    # unless field is the default field in which case is just {subfield}_dict
    # this is useful for reducing the size of dicts and strings generated from them

    # besides a name, each value in the dict should be a tuple consisting of a unique prefix to be found in the
    # string value and any casting function that converts a substring to some value
    # format: '{field}_{subfield}_dict' = dict(name1=(prefix_id1, type), name2=...)
    # value string as {prefix_id1}{value1}{SUBFIELD_DELIMITER}{prefix_id2}{value2}
    # e.g. field = eval and subfield = mode
    # EVAL_MODE_DICT = {subfield1: ('w', float), subfield2: ('z', str)} (or eval_mode_dict)
    # original dict : {eval_mode : 'w17.1,zoriginal'} =>
    # self[eval][subfield1] = 17.1 and self[eval][subfield2] = original
    # note that in general, instead of float or str, it can be any function that takes the substring and transforms it
    # to some other value:
    # EVAL_MODE_DICT = {subfield1: ('w', func1)} -> func1('17.1')
    # in general it can be some other subfield, e.g. type
    # EVAL_TYPE_DICT = ... , if the arg is provided as eval_type and the value is a string, then it will try to
    # fetch the corresponding parse dict and use it to generate additional subfields, see parse_if_defined

    # 3.
    # for properties, custom setters can also be defined and this is triggered during setitem

    # DUMPING THE CONTENTS FROM DICT TO A MORE COMPACT STR
    EXCLUDED_SUBFIELDS_IF_DEFAULT_VALUE = True

    def __init__(self, fields: Iterable[str] = None, default_field: str = None,
                 field_delimiter: str = None, attr_delimiter: str = None,
                 subfield_delimiter: str = None, immutable_after_loading: bool = None):
        object.__setattr__(self, '_in_init', True)
        self._cls_attrs = self.get_cls_attrs()
                           # {k: v for k, v in self.__class__.__dict__.items() \
                           # if not k.startswith('__') and not callable(v) and \
                           # not isinstance(v, classmethod)}
        self._has_loaded = False
        self._has_argparsed = False
        # the parsed dict from commands is stored in self.args, this will always remain unchanged after argparsing
        self._args = None
        # command history is stored in self.commands
        self._commands = {}
        # keeps track of the order of loaded keys in self (see argload and _update)
        self._loaded_keys = []
        AttributeFieldDict.__init__(self, fields=fields, default_field=default_field, field_delimiter=field_delimiter,
                                    attr_delimiter=attr_delimiter)
        self._subfield_delimiter = self.default_subfield_delimiter if subfield_delimiter is None else subfield_delimiter
        self.raise_error_if(not isinstance(self.sd, str))
        # None fetches the default
        self._immutable_after_loading = self.IMMUTABLE_AFTER_LOADING if immutable_after_loading is None else \
            immutable_after_loading
        self._in_init = False

    @property
    def reference_dict(self):
        return self.args   # for finding keys to exclude, iterate over args and not self!

    @property
    def args(self):
        return self._args

    @property
    def commands(self):
        return self._commands

    @property
    def subfield_delimiter(self):
        return self._subfield_delimiter

    @property
    def sd(self):
        return self._subfield_delimiter

    @property
    def has_argparsed(self):
        return self._has_argparsed and isinstance(self.args, dict)

    @property
    def has_loaded(self):
        return self._has_loaded and all([ak in self.loaded_keys for ak in self.args])

    @property
    def loaded_keys(self):
        return self._loaded_keys

    @property
    def immutable_after_loading(self):
        return self._immutable_after_loading

    @property
    def is_immutable(self): # locks all setters and throws error when trying to set
        return self.immutable_after_loading and self.has_loaded

    @property
    def excluded_keys(self):
        return ('name')

    def excluded_subfields_if(self, key, value):
        # if certain subfields are to be excluded, do so based on whether they have the same default values as
        # those given initially in commands
        return self.excluded_subfields_if_default_value and\
               isinstance(self.commands, dict) and self.commands[key].get('default') == value

    #############################################################################################
    #############################################################################################
    #############################################################################################
    # PARSING
    # PARSING FIRST ROUND: DEFINE COMMANDS USING ARGPARSE PKG
    # OVERRIDE THESE, as noted there can be multiple rounds of parsing here, as long as a next getter
    # in get_commands_extra is specified. This is useful for conditional parsing.
    @property
    def default_commands(self) -> Dict[str, Dict]:
        return dict(path=dict(type=str), name=dict(type=str))

    @property
    def require_commands_extra(self) -> bool:
        # depending on the values stored in self.args, one might need extra commands
        # in this case, get_commands_extra is run initially, it must return tuple containing extra commands
        # and the next getter. The conditional chain in parse stops when the getter is None.
        return False

    def get_commands_extra(self) -> Tuple[Dict[str, Dict], Union[None, Callable]]:
        # implement here conditional commands based on the default commands
        # make sure all extra commands are consumed, otherwise raise some error
        raise ValueError
    #############################################################################################

    @classmethod
    def argparse_with(cls, commands: Dict[str, Dict], args: List[str] = None, parse_all: bool = False,
                      description: str = None) -> Tuple[Dict, List[str]]:
        parser = argparse.ArgumentParser(description=description)
        for k, k_options in commands.items():
            parser.add_argument(f'-{k}', **k_options)  # NOTE: all are optional with '-'
        if parse_all:
            return vars(parser.parse_args(args))
        else:
            args, extras = parser.parse_known_args(args)
            return vars(args), extras

    def argparse(self, commands: Dict[str, Dict] = None, args: List[str] = None, parse_all: bool = True,
                 return_extras = True):
        # 1: parse args using commands via argparse
        self._has_argparsed = False
        if self.loaded_keys:  # reset loaded keys
            self._has_loaded = False
            for k in self.loaded_keys:
                del self[k]
            self._loaded_keys = []
        if commands is None:
            commands = self.default_commands
        self._commands.update(commands)
        self._args, extras = self.argparse_with(commands, args=args)
        if parse_all and (self.require_commands_extra or extras):
            get_commands_extra = self.get_commands_extra
            while get_commands_extra:  # keep looping on new conditional commands, if there is such chain
                commands_extra, get_commands_extra = get_commands_extra()
                self._commands.update(commands_extra)
                if get_commands_extra:
                    args, extras = self.argparse_with(commands_extra, args=extras)
                    self._args.update(args)
                else:  # consume everything if its the last and parse_all is true
                    args = self.argparse_with(commands_extra, args=extras, parse_all=True)
                    self._args.update(args)
                    extras = ()
                    break
        self._has_argparsed = True
        return self.args, extras if return_extras else self.args

    #############################################################################################
    # ADDITIONAL PARSING TRIGGERED INSTANTLY WHEN SETTING SELF ITEMS

    def argload(self, commands: Dict[str, Dict] = None, args: List = None, parse_all: bool = True,
             keys: Union[str, Iterable[str]] = None,
             load_all_remaining: bool = False, return_remaining: bool = False,
             return_extras: bool = False):
        if not self.has_loaded: # just to be safe, don't run if it loaded all keys, create a new object if needed
            extras = self.argparse(commands=commands, args=args, parse_all=parse_all, return_extras=True)[1] \
                if not self.has_argparsed else ()
            # while loading args
            # 2: additional parsing of args when setting self items using parse dicts defined as class
            # or instance attributes, and 3: property setters
            arg_keys = self.args.keys()
            if keys is None:  # if keys to load are not given, assume all from args (this is the order of the update)
                keys = arg_keys
            else:
                keys = [keys] if isinstance(keys, str) else list(keys)
                keys_in_arg = [k for k in keys if k in arg_keys] # only keep those in arg_keys
                self.raise_error_if(len(keys) != len(keys_in_arg), error=RuntimeError,
                                    msg='Loading error: Not all keys in arg keys.')
                if load_all_remaining:
                    keys.append([k for k in arg_keys if k not in keys])
            for k in keys: # arg loading only in these two lines, rest is bookkeeping
                self[k] = self.args[k]
            # check whether keys have been correctly loaded and that keys in self include keys in args
            remaining_keys = [ak for ak in arg_keys if ak not in self.loaded_keys]
            self._has_loaded = len(remaining_keys) == 0
            output = remaining_keys if return_remaining else self
            output = (output, extras) if return_extras else output
            return output

    @classmethod
    def value_from_str(cls, txt, prefix, type=str, delimiter=None):
        if delimiter is None:
            delimiter = cls.DEFAULT_SUBFIELD_DELIMITER
        return AttributeFieldDict.value_from_str(txt=txt, prefix=prefix, type=type, delimiter=delimiter)

    @classmethod
    def replace_value_in_str(cls, txt, value, prefix, delimiter=None):
        if delimiter is None:
            delimiter = cls.DEFAULT_SUBFIELD_DELIMITER
        return AttributeFieldDict.replace_value_in_str(txt=txt, value=value, prefix=prefix, delimiter=delimiter)

    def is_parse_defined(self, key):
        return self.to_parse_dict_name(key) in self.cls_attrs

    def get_parse_options(self, key, name=None):
        parse_dict = getattr(self, self.to_parse_dict_name(key))
        return  parse_dict[name] if name else parse_dict

    def to_parse_dict_name(self, key, to_upper=True):
        pd_name =f'{self.to_attr_name(key)}_dict'
        return pd_name.upper() if to_upper else pd_name

    def replace_named_value_in_subfield(self, txt, key, name, value):
        p, t = self.get_parse_options(key=key, name=name)
        txt = self.replace_value_in_str(txt, value, p, self.sd)
        self[key] = txt
        return txt

    # Updating setattr with an extra check regarding immutability. Given the size of the function,
    # it is best to simply override it as the bulk is done in the new _update
    def __setattr__(self, name: str, value: Any):
        # print("IN SETATTR")
        # print(name, value)
        if name.startswith('_') or self._in_init:  # handle private attributes normally
            self.__dict__[name] = value
        elif name in self.__dict__:  # can only set existing normal attributes if not immutable
            self.raise_error_if(self.is_immutable, error=RuntimeError, msg='Immutable.')
        else:
            field, subfield = self.attr_to_field_subfield(name)
            self._update(field=field, subfield=subfield, value=value)

    def __delitem__(self, key: str):
        self.raise_error_if(self.is_immutable, error=RuntimeError, msg='Immutable.')
        super().__delitem__(key)


    def _update(self, key: str = None, field: str = None, subfield: str = None, value: Any = None):
        # print(key, field, subfield, value)
        # TODO: perhaps add a check here such that only arg keys are allowed, but might be too restrictive
        self.raise_error_if(self.is_immutable, error=RuntimeError, msg='Immutable.')
        # if a key belongs to the default field, the attribute name should not have the field in its name
        # e.g., default_field = base, then for subfield = path, name the property as path and not base_path,
        # otherwise, the field should be included, separated by the corresponding attr delimiter
        updated_attrs = {}
        # 2nd round: parse dicts defined as class or instance attributes: field_subfield_dict
        if key is None:
            key = self.to_key(field=field, subfield=subfield)
        if isinstance(value, str):
            if self.is_parse_defined(key):  # add more entries by parsing txt
                txt = value
                parse_dict = self.get_parse_options(key)
                field, subfield = self.to_field_subfield(key)
                new_value = ''
                # print("PARSE DICT: ", parse_dict)
                # print(f"FIELD:{field}, SUBFIELD:{subfield}")
                prefixes = [v[0] for v in parse_dict.values()]
                for ri, (k, options) in enumerate(parse_dict.items()):  # p, t = v
                    p, t = options[:2]
                    prefixes.remove(p)
                    new_txt = txt
                    for pp in prefixes:
                        if pp.startswith(p): # invalidate other subfields that share same pre-prefix
                             new_txt = new_txt.replace(pp, INVALID_PREFIX)
                    if any([st.startswith(p) for st in new_txt.split(self.sd)]):  # if identifier in txt
                        txt_to_parse = new_txt
                    else: # get default txt to parse
                        try:
                            txt_to_parse = getattr(self, f'default{self.ad}{key}')
                        except AttributeError:
                            raise AttributeError(f'For {field}{self.ad}{k}, {p} not in {txt} and'
                                                 f'default txt not defined as default_{field}{self.ad}{key}'
                                                 f' (upper or lower). Mismatch parse dict and default value (txt).')
                    # NOTE: self[field] is just a dict, hence not subject to the same setitem rules
                    # print(k, ': ', txt_to_parse, t, p)
                    subvalue = self.value_from_str(txt=txt_to_parse, prefix=p, type=t, delimiter=self.sd)
                    # TODO: a more elegant way of doing this
                    len_options = len(options)
                    # SUBVALUE CHECKING AND GENERATION
                    if len_options == 3:
                        condfn = options[2]
                        if isinstance(condfn, str):
                            condfn =  getattr(self, condfn)
                        self.raise_error_if(not condfn(subvalue), msg=self.invalid_value_message(field, k, subvalue))
                    elif len_options == 4:
                        condfn, (genfn, required_attrnames) = options[2:4]
                        if isinstance(condfn, str):
                            condfn = getattr(self, condfn)
                        if isinstance(genfn, str):
                            genfn = getattr(self, genfn)
                        try:                      # default error is ValueError, just making sure if it ever changes
                            self.raise_error_if(not condfn(subvalue), error=ValueError,
                                                msg=self.invalid_value_message(field, k, subvalue))
                        except ValueError:
                            # print({a: getattr(self, a) for a in required_attrnames})
                            subvalue = genfn(**{a: getattr(self, a) for a in required_attrnames})
                            self.raise_error_if(not condfn(subvalue),
                                                msg=self.invalid_value_message(field, k, subvalue))
                    elif len_options > 4:
                        raise NotImplementedError
                    self[field][k] = subvalue
                    new_value += (p + (str(subvalue).lower() if isinstance(subvalue, bool) else str(subvalue)) + self.sd)
                    updated_attrs[self.field_subfield_to_attr_name(field, k)] = subvalue
                value = new_value.rstrip(self.sd)
        # update original entry last, this is useful when the original setter requires all other additional to be
        # run first
        field, subfield = self.to_field_subfield(key)
        self[field][subfield] = value
        updated_attrs[self.field_subfield_to_attr_name(field, subfield)] = value
        # bookkeeping of loaded keys
        if key in self._loaded_keys:  # avoid duplicates and keep order of last update in loaded keys
            self._loaded_keys.remove(key)
        self._loaded_keys.append(key)
        # 3rd round: call a property setter if its defined
        # NOTE: values are put first in the (inner) dict, and only then the setters are called,
        # so that if one does not want to modify the values further, there is no need to define a custom setter
        # it only makes sense to handle further processing of values as well as conditional setting of other
        # items
        # the name of the property should match key up to the respective delimiter (by default they are the same, '_')
        # IMPORTANT: BE CAREFUL NOT TO CALL THE SAME SET ITEM IN A PROPERTY SETTER,
        # use self[field][subfield] = ...,
        # and NOT self[key] = ...
        # print("UPDATED: ", updated_attrs)
        for name, value in updated_attrs.items():
            if name in self.property_setter_names:
                # print(name, self.property_setters[name])
                self.property_setters[name](self, value)
        # print("==== OUT OF UPDATE ======")
        return self

    def dump(self):
        # TODO: make this auto
        d = super().dump()
        d.update({'_args': self._args, '_commands': self._commands, '_subfield_delimiter': self._subfield_delimiter,
                  '_has_argparsed': self._has_argparsed,
                  '_has_loaded': self._has_loaded, '_loaded_keys': self._loaded_keys,
                  '_immutable_after_loading': self._immutable_after_loading})
        return d