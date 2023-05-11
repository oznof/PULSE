import os
import dill
import shutil
from glob import glob
from pathlib import Path

import numpy as np

from ..caster import list_unique, list_remove, to_list, to_list_str, to_sized_list, to_item_if_singleton
from ..errors import raise_error_if


class IO:
    @classmethod
    def fix_path(cls, value):
        return value.replace('~', str(Path.home()))

    @classmethod
    def join_path(cls, path, *args):
        return os.path.join(path, *args)

    @classmethod
    def split_path(cls, path, ext=None):
        path, name = os.path.split(path)
        if ext:
            name = cls.file_name(name, split=False, ext=ext)
        return path, name

    @classmethod
    def parent_dir(cls, path):
        return os.path.basename(os.path.dirname(path))

    @classmethod
    def split_ext(cls, file):
        return os.path.splitext(file)

    @classmethod
    def get_name(cls, file, split=True, ext=None):
        name = cls.split_path(file)[-1] if split else file
        if ext:
            name = name.rstrip(ext if ext.startswith('.') else f'.{ext}')
        return name

    @classmethod
    def makedir(cls, path=None):
        if not isinstance(path, str):
            return
        path = cls.fix_path(path)
        if not os.path.exists(path):
            os.makedirs(path)

    @classmethod
    def file_name(cls, file, split=True, ext=None):
        name = cls.split_path(file)[-1] if split else file
        if ext:
            name = name.rstrip(ext if ext.startswith('.') else f'.{ext}')
        return name

    @classmethod
    def isfile(cls, file):
        return os.path.isfile(file)

    @classmethod
    def move_file(cls, source, target):
        cls.makedir(os.path.dirname(target))
        shutil.move(cls.fix_path(source), cls.fix_path(target))
        return target

    @classmethod
    def safe_remove_file(cls, file):
        file = cls.fix_path(file)
        if os.path.exists(file):
            os.remove(file)

    @classmethod
    def list_files(cls, path, name='*'):
        return glob(os.path.join(cls.fix_path(path), name))

    @classmethod
    def file_ext(cls, file):
        ext = cls.split_ext(file)[-1].lstrip('.')
        if ext not in ['pkl', 'dill', 'txt']:
            ext = None
        return ext

    @classmethod
    def add_ext(cls, name, ext='txt'):
        if not name.endswith(f'.{ext}'):
            name += f'.{ext}'
        return name

    @classmethod
    def load_file(cls, file, **kwargs):
        ext = cls.file_ext(file)
        if ext is None or ext in ['pkl', 'dill']:
            return cls.load_dill(file, **kwargs)
        elif ext == 'txt':
            return cls.load_txt(file, **kwargs)
        else:
            raise NotImplementedError(f'{ext} is not a valid data loading ext.')

    @classmethod
    def load_dill(cls, file, **kwargs):
        file = cls.fix_path(file)
        with open(file, 'rb') as dill_file:
            data = dill.load(dill_file)
        return data

    @classmethod
    def load_txt(cls, file, dtype=str, delimiter='\n', **kwargs):
        file = cls.fix_path(file)
        data = np.loadtxt(cls.add_ext(file, ext='txt'), dtype=dtype, delimiter=delimiter)
        if dtype == str:
            data = to_list_str(data.tolist())
        return data

    @classmethod
    def save_file(cls, file, data, **kwargs):
        ext = cls.file_ext(file)
        if ext is None or ext in ['pkl', 'dill']:
            cls.save_dill(file, data, **kwargs)
        elif ext == 'txt':
            cls.save_txt(file, data, **kwargs)
        else:
            raise NotImplementedError(f'{ext} is not a valid data saving ext.')

    @classmethod
    def save_dill(cls, file, data, **kwargs):
        file = cls.fix_path(file)
        cls.makedir(os.path.dirname(file))
        with open(file, 'wb') as dill_file:
            dill.dump(data, dill_file)

    @classmethod
    def save_txt(cls, file, data, append=False, fmt='%s', delimiter='\n', **kwargs):
        file = cls.fix_path(file)
        cls.makedir(os.path.dirname(file))
        if not isinstance(data, np.ndarray):
            data = np.array(to_list(data))
        open_mode = 'ab' if append else 'w'
        with open(cls.add_ext(file, ext='txt'), open_mode) as f:
            np.savetxt(f, data, fmt=fmt, delimiter=delimiter)

    @classmethod
    def update_file(cls, file, data, **kwargs):
        ext = cls.file_ext(file)
        if ext is None or ext in ['pkl', 'dill']:
            return cls.update_dill(file, data, **kwargs)
        elif ext == 'txt':
            return cls.update_txt(file, data, **kwargs)
        else:
            raise NotImplementedError(f'{ext} is not a valid data updating ext.')

    @classmethod
    def update_dill(cls, file, data, **kwargs):
        raise_error_if(not isinstance(data, dict), msg="Data must be a dict.")
        if not os.path.isfile(cls.fix_path(file)):
            cls.save_dill(file, data, **kwargs)
        else:
            new_data = cls.load_dill(file, **kwargs)
            new_data.update(data)
            cls.save_dill(file, new_data, **kwargs)

    @classmethod
    def update_txt(cls, file, data, unique=False, sort=False, **kwargs):
        if not os.path.isfile(cls.add_ext(cls.fix_path(file), ext='txt')):
            cls.save_txt(file, data, **kwargs)
        else:
            old = cls.load_txt(file)
            if isinstance(old, list):
                new = to_list(old) + to_list(data)
            elif isinstance(old, np.ndarray):
                new = np.concatenate((old, np.atleast_1d(np.array(data))))
            else:
                raise TypeError("Invalid data type.")
            if unique:
                new = list_unique(new) if isinstance(new, list) else np.unique(new)
            if sort:
                new = sorted(new) if isinstance(new, list) else np.sort(new)
            cls.save_txt(file, new)

    @classmethod
    def remove_file(cls, file, data, **kwargs):
        ext = cls.file_ext(file)
        if ext is None or ext in ['pkl', 'dill']:
            return cls.remove_dill(file, data, **kwargs)
        elif ext == 'txt':
            return cls.remove_txt(file, data, **kwargs)
        else:
            raise NotImplementedError(f'{ext} is not a valid data updating ext.')

    @classmethod
    def remove_txt(cls, file, data, **kwargs):
        exists = False
        file = cls.add_ext(cls.fix_path(file), ext='txt')
        if os.path.isfile(file):
            new = cls.load_txt(file, **kwargs)
            new = list_remove(new, data)
            exists = bool(new)
            cls.save_txt(file, new, **kwargs) if exists else os.remove(file)
        return exists

    @classmethod
    def remove_dill(cls, file, data, **kwargs):
        exists = False
        file = cls.fix_path(file)
        if os.path.isfile(file):
            new = cls.load_dill(file, **kwargs)
            [new.pop(k) for k in to_list(data)]
            exists = bool(new)
            cls.save_dill(file, new, **kwargs) if exists else os.remove(file)
        return exists

    @classmethod
    def is_in_file(cls, file, data, **kwargs):
        ext = cls.file_ext(file)
        if ext is None or ext in ['pkl', 'dill']:
            return cls.is_in_dill(file, data, **kwargs)
        elif ext == 'txt':
            return cls.is_in_txt(file, data, **kwargs)
        else:
            raise NotImplementedError(f'{ext} is not a valid ext.')

    @classmethod
    def is_in_txt(cls, file, data, **kwargs):
        file = cls.add_ext(cls.fix_path(file), ext='txt')
        data = to_list(data)
        if not os.path.isfile(file):
            return to_item_if_singleton(to_sized_list(False, len(data)))
        stored_data = cls.load_txt(file, **kwargs)
        is_in = to_item_if_singleton([d in stored_data for d in data])
        return is_in

    @classmethod
    def is_in_dill(cls, file, data, **kwargs):
        data = to_list(data)
        if not os.path.isfile(cls.fix_path(file)):
            return to_item_if_singleton(to_sized_list(False, len(data)))
        stored_data = cls.load_dill(file, **kwargs)
        is_in = to_item_if_singleton([d in stored_data for d in data])
        return is_in

    @classmethod
    def show_sep(cls, sep='=', times=100):
        print(sep * times)