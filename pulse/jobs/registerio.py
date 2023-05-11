import os
from datetime import datetime

from ..caster import to_list_str, list_unique, to_item_if_singleton, to_sized_list
from .io import IO


class RegisterIO(IO):
    PKG_NAME = 'PKG'
    DEFAULT_PATH = f'~/storage/data/{PKG_NAME}/'
    DEFAULT_FILE_NAME = 'job'
    DEFAULT_REGISTER_SUBPATH = 'register'
    DEFAULT_REGISTER_NAME = 'general'
    REGISTER_KEYWORD = None

    def __init__(self, path=None, name=None, file=None, register=None, overwrite=False):
        self.path = path
        self.name = name
        self.file = file
        self.register = register
        self.overwrite = overwrite

    @property
    def path(self):
        return self._path

    @path.setter
    def path(self, value):
        if value is None:
            value = self.DEFAULT_PATH
        value = self.fix_path(value)
        self._path = value

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, value):
        if value is None:
            value = f"{self.DEFAULT_FILE_NAME}_{datetime.now().strftime('%d-%m-%Y_%H:%M:%S')}"
        self._name = value

    @property
    def file(self):
        return self._file

    @file.setter
    def file(self, value):
        if value is None:
            value = os.path.join(self.path, self.name)
        else:
            # make sure path and name are updated to same values
            path, name = self.split_path(value)
            if path != self.path:
                self.path = path
            if name != self.name:
                self.name = name
        self._file = value

    @property
    def register(self):
        return self._register

    @register.setter
    def register(self, value):
        if value is None:
            path = os.path.join(self.path.split(self.PKG_NAME, 1)[0], self.PKG_NAME,
                                self.DEFAULT_REGISTER_SUBPATH)
            if self.REGISTER_KEYWORD and self.REGISTER_KEYWORD in self.name:
                value = self.name.split(self.REGISTER_KEYWORD, 1)[1].lstrip().split(' ', 1)[0]
            else:
                value = self.DEFAULT_REGISTER_NAME
            value = os.path.join(path, value)
        if not value.endswith('.txt'):
            value += '.txt'
        self._register = value

    @property
    def file_exists(self):
        return os.path.isfile(self.file)

    @property
    def register_exists(self):
        return os.path.isfile(self.register)

    @property
    def in_register(self):
        return self.is_in_register()

    def coerce_register_types(self, register=None, files=None, unique=True, sort=True):
        if not register:
            register = self.register
        register = self.fix_path(register)
        if files is None:
            return register
        files = to_list_str(files)
        if unique:
            files = list_unique(files)
        if sort:
            files = sorted(files)
        return register, files

    def load_register(self, register=None, **kwargs):
        register = self.coerce_register_types(register)
        return self.load_txt(register, **kwargs)

    def save_register(self, files, register=None, unique=True, sort=True, **kwargs):
        register, files = self.coerce_register_types(register=register, files=files,
                                                     unique=unique, sort=sort)
        self.save_txt(register, files, **kwargs)

    def update_register(self, files=None, register=None, unique=True, sort=True, **kwargs):
        if not files:
            files = self.file
        register, files = self.coerce_register_types(register=register, files=files,
                                                     unique=unique, sort=sort)
        if unique:
            # this removes files
            self.remove_register(files, register, unique=True, sort=sort, **kwargs)
        self.update_txt(register, files, unique=unique, sort=sort, **kwargs)

    def remove_register(self, files=None, register=None, unique=True, sort=True, coerce_inputs=True, **kwargs):
        if coerce_inputs:
            if not files:
                files = self.file
            register, files = self.coerce_register_types(register=register, files=files,
                                                         unique=unique, sort=sort)
        if not os.path.isfile(register):
            return False
        if not unique:
            return self.remove_txt(register, files, **kwargs)
        register_files = self.load_register(register, **kwargs)
        new = []
        paths, names = zip(*[self.split_path(file) for file in files])
        for rfile in register_files:
            rpath, rname = self.split_path(rfile)
            if rname not in names: # only keep files whose name does not match
                if rfile not in new:
                    new.append(rfile)
            elif rfile not in files: # remove rfile if shares a name, but not listed
                self.safe_remove_file(rfile)
        exists = bool(new)
        self.save_register(new, register, sort=sort, **kwargs) if exists else os.remove(register)
        return exists

    def is_in_register(self, files=None, register=None, unique=True, **kwargs):
        if not files:
            files = self.file
        register, files = self.coerce_register_types(register=register, files=files,
                                                     unique=False, sort=False)
        if os.path.isfile(register):
            if self.overwrite:
                self.remove_register(files, register, coerce_inputs=False, **kwargs)
            else:
                if unique: # compare names else include paths in the comparison
                    rfiles = self.load_register(register, **kwargs)
                    rfiles = [self.get_name(rfile) for rfile in rfiles]
                    return to_item_if_singleton([self.get_name(file) in rfiles for file in files])
                return self.is_in_txt(register, files, **kwargs)
        return to_item_if_singleton(to_sized_list(False, len(files)))

    def load(self, **kwargs):
        return self.load_file(self.file, **kwargs)

    def save(self, data,**kwargs):
        self.save_file(self.file, data, **kwargs)
        self.update_register(self.file, **kwargs)

    def update(self, data, **kwargs):
        self.update_file(self.file, data, **kwargs)
        if not self.in_register:
            self.update_register(self.file, **kwargs)

    def remove(self, data=None, **kwargs):
        if data is None:
            # remove file if exists, and remove it from register, output existence of register
            self.safe_remove_file(self.file)
            exists = self.remove_register(self.file, **kwargs)
        else:
            # remove data, and if it no longer exists, remove it from register, output existence of file
            exists = self.remove_file(self.file, data, **kwargs)
            if not exists:
                self.remove_register(self.file, **kwargs)
        return exists

    def is_in(self, files=None, data=None, register=None, **kwargs):
        if data is None:
            return self.is_in_register(files=files, register=register, **kwargs)
        return self.is_in_file(self.file, data, **kwargs)