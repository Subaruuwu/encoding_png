import time


def get_time_func(func):
    def wrapper(*args):
        t0 = time.time()
        result = func(*args)
        t1 = time.time()
        return t1 - t0, result
    return wrapper
