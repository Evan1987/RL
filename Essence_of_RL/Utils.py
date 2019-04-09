"""
Some Common Tools Collected from Other Scripts
"""

import time
from contextlib2 import contextmanager


@contextmanager
def timer(name, verbose: bool=True):
    """
    模块执行时间计算的上下文函数
    :param name: 模块名称
    :param verbose:  是否打印结果
    """
    start = time.time()
    try:
        yield
    finally:
        end = time.time()
        if verbose:
            print("%s Time Cost: %.6f" % (name, end - start))
