import os

from .registerio import RegisterIO


class PendingIO(RegisterIO):
    DEFAULT_PENDING_NAME = 'pending'

    def __init__(self, path=None, name=None, file=None, register=None, pending=None, overwrite=False):
        super().__init__(path=path, name=name, file=file, register=register, overwrite=overwrite)
        self.pending = pending

    @property
    def pending(self):
        return self._pending

    @pending.setter
    def pending(self, value):
        if value is None:
            value = os.path.dirname(self.register)
        if self.split_path(value)[0]:
            value = os.path.join(value, self.DEFAULT_PENDING_NAME)
        if not value.endswith('.txt'):
            value += '.txt'
        self._pending = value

    @property
    def pending_exists(self):
        return os.path.isfile(self.pending)

    @property
    def in_pending(self):
        return self.is_in_pending()

    @property
    def start(self):
        #if self.overwrite:
        #    self.remove()
        if not self.overwrite and (self.in_pending or self.in_register):
            return False
        self.update_pending(self.file)
        return True

    def load_pending(self, **kwargs):
        return self.load_register(register=self.pending, **kwargs)

    def save_pending(self, *args, **kwargs):
        self.save_register(*args, register=self.pending, **kwargs)

    def update_pending(self, *args, **kwargs):
        self.update_register(*args, register=self.pending, **kwargs)

    def remove_pending(self, *args, **kwargs):
        return self.remove_register(*args, register=self.pending, **kwargs)

    def is_in_pending(self, files=None, **kwargs):
        return self.is_in_register(files=files, register=self.pending, **kwargs)

    def save(self, data, pending=True, **kwargs):
        self.save_file(self.file, data, **kwargs)
        if pending:
            self.update_pending(self.file, **kwargs)
        else:
            self.remove_pending(self.file **kwargs)
            self.update_register(self.file, **kwargs)

    def update(self, data, pending=True, **kwargs):
        self.update_file(self.file, data, **kwargs)
        if pending:
            if not self.in_pending:
                self.update_pending(self.file, **kwargs)
        else:
            if not self.in_register:
                self.update_register(self.file, **kwargs)
            self.remove_pending(self.file, **kwargs)

    def remove(self, data=None, **kwargs):
        exists = super().remove(data=data, **kwargs)
        if not self.file_exists:
            self.remove_pending(self.file, **kwargs)
        return exists

    def is_in(self, files=None, data=None, register=None, pending=False, **kwargs):
        if pending:
            return self.is_in_pending(files=files, **kwargs)
        return super().is_in(files=files, data=data, register=register, **kwargs)

    def end(self, data, **kwargs):
        self.update(data, pending=False, **kwargs)