# -*- coding: utf-8 -*-
from __future__ import print_function, division
import threading, sys, numpy as np, time
try:
    # Python 3
    import builtins
except ImportError:
    # Python 2
    import __builtin__ as builtins

def print(*args, **kwargs):
    sep, end = kwargs.pop('sep', ' '), kwargs.pop('end', '\n')
    file, flush = kwargs.pop('file', sys.stdout), kwargs.pop('flush', False)
    if kwargs:
        raise TypeError('print() got an unexpected keyword argument {!r}'.format(next(iter(kwargs))))
    builtins.print(*args, sep=sep, end=end, file=file)
    if flush:
        file.flush()

#==============================================================================

default_spinner = 'dqpb'

#==============================================================================

class StatusLine:
    def __init__(self, msg, print_time=True, spinner=None, callback=None, timer=True):
        self.msg = msg
        self.print_time = print_time
        self.spinner = spinner
        self.callback = callback
        self.timer = timer
    def __enter__(self, *args):
        print("{}... ".format(self.msg), end="", flush=True)
        self.thread = StatusThread(spinner=self.spinner, callback=self.callback, timer=self.timer)
        self.thread.start()
        self.st = time.time()
    def __exit__(self, *args):
        self.thread.stop()
        self.thread.join()
        if self.print_time: print("done. {:.4f} seconds elapsed.".format(time.time()-self.st), flush=True)
        else              : print("done.", flush=True)

class UnsupportedSpinnerException(Exception):
    def __init__(self, msg): super(UnsupportedSpinnerException, self).__init__(msg)

#------------------------------------------------------------------------------

def set_default_spinner(s):
    global default_spinner
    default_spinner = s

#------------------------------------------------------------------------------

class StatusThread(threading.Thread):
    def __init__(self, spinner, callback, timer):
        if spinner is None and default_spinner is None:
            spinner = list(Spinners.keys())[np.random.randint(len(Spinners.keys()))]
        elif spinner is None and default_spinner is not None:
            spinner = default_spinner
        if spinner not in Spinners:
            raise UnsupportedSpinnerException("Unsupported spinner '{}'".format(spinner))
        self.spinner = Spinners[spinner]
        self._start_time = time.time(); self.timer = timer
        def make_frame(t, msg):
            frame = self.spinner["frames"][t%len(self.spinner["frames"])]
            if msg is not None: frame = "{} {}".format(msg, frame)
            if self.timer: frame = "{:.01f}s {}".format(time.time()-self._start_time, frame)
            return frame
        def print_progress():
            self.timestep = 0
            if callback is not None: frame = make_frame(self.timestep, callback())
            else                   : frame = make_frame(self.timestep, None)
            print("{} ".format(frame), sep="", end="", flush=True)
            at_least_once = False
            while not at_least_once or not self.stopped():
                backspaces = "\b"*(len(frame)+1)
                self.timestep += 1
                if callback is not None: frame = make_frame(self.timestep, callback())
                else                   : frame = make_frame(self.timestep, None)
                print("{}{} ".format(backspaces, frame), sep="", end="", flush=True)
                self._stop_event.wait(self.spinner["interval"]/1000)
                at_least_once = True
            backspaces = "\b"*(len(frame)+1)
            print("{}".format(backspaces), sep="", end="", flush=True)
        super(StatusThread, self).__init__(target=print_progress)
        self._stop_event = threading.Event()
    def stop(self): self._stop_event.set()
    def stopped(self): return self._stop_event.is_set()

#------------------------------------------------------------------------------

Spinners = {
        "dqpb": {
                "interval": 100,
                "frames": [
                        "d",
                        "q",
                        "p",
                        "b"
                ]
        },
}
