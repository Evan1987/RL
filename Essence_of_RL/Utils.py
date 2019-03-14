"""
Some Common Tools Collected from Other Scripts
"""

import time
from contextlib2 import contextmanager


@contextmanager
def timer(name, verbose: int=1):
    """
    模块执行时间计算的上下文函数
    :param name: 模块名称
    :param verbose:  是否打印结果
    """
    start = time.perf_counter()
    try:
        yield
    finally:
        end = time.perf_counter()
        if verbose:
            print("%s COST: %.6f" % (name, end - start))