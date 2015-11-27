# -*- coding: utf-8 -*-
from itertools import chain


def flatten(y):
    """
    Flatten a list of lists.

    >>> flatten([[1,2], [3,4]])
    [1, 2, 3, 4]
    """
    return list(chain.from_iterable(y))
