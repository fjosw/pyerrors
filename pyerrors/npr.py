import numpy as np


class Propagator(np.ndarray):

    def __new__(cls, input_array, mom=None):
        obj = np.asarray(input_array).view(cls)
        obj.mom = mom
        return obj

    def __array_finalize__(self, obj):
        if obj is None: return
        self.mom = getattr(obj, 'mom', None)
