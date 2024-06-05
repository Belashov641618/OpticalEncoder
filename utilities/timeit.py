from time import time


class Timings:
    _T0:float
    def __init__(self):
        self._T0 = time()
    def __call__(self, name:str=None):
        if name is None:
            self._T0 = time()
        else:
            print(f"{name}: {time() - self._T0}")
            self._T0 = time()
log_timings = Timings()