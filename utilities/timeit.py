from __future__ import annotations
import numpy
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

class Chronograph:
    def __init__(self):
        self._processes_names = []
        self._processes_time_start = []
        self._processes_timings = []

    _processes_names:list[str]
    def add_process(self, name:str) -> int:
        if name not in self._processes_names:
            id = len(self._processes_names)
            self._processes_names.append(name)
            self._processes_time_start.append(0)
            self._processes_timings.append([])
            return id
        else:
            return self._processes_names.index(name)

    _processes_time_start:list[float]
    def start(self, id:int):
        self._processes_time_start[id] = time()

    _processes_timings:list[list[float]]
    def end(self, id:int):
        self._processes_timings[id].append(time() - self._processes_time_start[id])

    def statistic(self):
        class ChronographStatisticSample:
            name:str
            mean:float
            error:float
            data:numpy.ndarray
            def __init__(self, name:str, data:numpy.ndarray):
                self.name = name
                self.data = data
                self.mean = numpy.mean(data)
                self.error = numpy.std(data) / numpy.sqrt(data.shape[0] - 1 if data.shape[0] >= 2 else 1)
        class ChronographStatistic:
            processes:list[ChronographStatisticSample]
            def __init__(self):
                self.processes = []
            def append(self, process:ChronographStatisticSample):
                self.processes.append(process)
            def print(self):
                for process in self.processes:
                    print(f"{process.name} time is {process.mean} with error {process.error}")
            def __add__(self, other:ChronographStatistic):
                processes = self.processes
                names = [process.name for process in self.processes]
                for process in other.processes:
                    if process.name in names:
                        id = names.index(process.name)
                        processes[id] = ChronographStatisticSample(process.name, numpy.stack((process.data, processes[id].data)))
                    else:
                        processes.append(process)
                new = ChronographStatistic()
                new.processes = processes
                return new
        statistic = ChronographStatistic()
        for process_timing, name_ in zip(self._processes_timings, self._processes_names):
            process_timing = numpy.array(process_timing)
            sample = ChronographStatisticSample(name_, process_timing)
            statistic.append(sample)
        return statistic