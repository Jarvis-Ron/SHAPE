"""SHAPE implementation for the fixed release pipeline.

Only dependencies of the retained SHAPE stages are included. See docs/code-map.md.
"""

import os
import numpy as np


class ProcessLocalNpzArchive:
    """Own one ``NpzFile`` handle per operating-system process.

    ``DataLoader`` fork workers inherit the parent's Python objects, including
    ``zipfile.ZipFile`` and its lock.  The file descriptor is shared across
    processes but the lock is not, so concurrent reads can corrupt the shared
    seek position and surface as ``BadZipFile: Overlapped entries``.  Reopen
    the archive on the first access in every process instead.
    """

    def __init__(self, path, **load_kwargs):
        self.path = os.fspath(path)
        self.load_kwargs = dict(load_kwargs)
        self._handle = None
        self._owner_pid = None

    def get(self):
        current_pid = os.getpid()
        if self._handle is None or self._owner_pid != current_pid:
            self.close()
            self._handle = np.load(self.path, **self.load_kwargs)
            self._owner_pid = current_pid
        return self._handle

    @property
    def owner_pid(self):
        return self._owner_pid

    def close(self):
        handle = self._handle
        self._handle = None
        self._owner_pid = None
        if handle is not None:
            handle.close()

    def __getstate__(self):
        # Spawn workers pickle the dataset instead of inheriting it.  Never
        # serialize a live ZipFile; the receiving process opens its own copy.
        state = self.__dict__.copy()
        state["_handle"] = None
        state["_owner_pid"] = None
        return state

    def __del__(self):
        try:
            self.close()
        except Exception:
            # Interpreter shutdown may already have torn down dependencies.
            pass
