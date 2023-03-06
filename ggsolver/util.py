import copy
import math
import numpy as np
from itertools import chain, combinations


class BColors:
    """
    # Reference: https://stackoverflow.com/questions/287871/how-do-i-print-colored-text-to-the-terminal
    """
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


class ColoredMsg:
    @staticmethod
    def ok(msg):
        return f"{BColors.OKCYAN}{msg}{BColors.ENDC}"

    @staticmethod
    def warn(msg):
        return f"{BColors.WARNING}{msg}{BColors.ENDC}"

    @staticmethod
    def success(msg):
        return f"{BColors.OKGREEN}{msg}{BColors.ENDC}"

    @staticmethod
    def error(msg):
        return f"{BColors.FAIL}{msg}{BColors.ENDC}"

    @staticmethod
    def header(msg):
        return f"{BColors.HEADER}{msg}{BColors.ENDC}"


class Distribution:
    def __init__(self, domain, prob):
        assert all(isinstance(element, int) for element in domain)
        assert np.isclose(sum(prob), 1.0)
        assert len(domain) == len(prob)

        self.domain = list(domain)
        self.prob = list(prob)

    def support(self):
        return [self.domain[i] for i in range(len(self.domain)) if self.prob[i] > 0]

    def pmf(self, element):
        return self.prob[self.domain.index(element)]
def apply_atoms_limit(atoms):
    if len(atoms) > 16:
        raise ValueError("ggsolver supports atoms set up to size 16.")


def powerset(iterable):
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))


def cantor_pairing(tuple_):
    """
    Cantor pairing function maps an n-dimensional tuple of positive integers to a single positive integer.
    The mapping is invertible.

    :param tuple_: (Iterable of int) List/tuple of positive integers.
    :return: (int) Single integer.

    .. note:: Modified from https://github.com/perrygeo/pairing/blob/master/pairing/main.py
    """
    n = len(tuple_)
    if n == 0:
        return 0
    elif n == 1:
        return tuple_[0]
    else:
        e0 = cantor_pairing(tuple_[:-1])
        e1 = tuple_[-1]
        out = int(0.5 * (e0 + e1) * (e0 + e1 + 1) + e1)
        if (e0, e1) != inverse_cantor_pairing(out, d=2):
            raise ValueError(f"{e0} and {e1} could not be cantor-paired.")
        return out


def inverse_cantor_pairing(n, d):
    """
    Inverse cantor pairing function maps a single positive integer to an n-dimensional tuple of positive integers.
    The mapping is invertible.

    :param n: (int) The number to be decoupled.
    :param d: (int) Dimension of tuple to be constructed.
    :return: (tuple) n-dimensional tuple of positive integers.

    .. note:: Modified from https://github.com/perrygeo/pairing/blob/master/pairing/main.py
    """
    # Minimum dimension is 1.
    assert d > 0

    # If input is a number, initialize it as tuple.
    if isinstance(n, int):
        n = [n]

    # 1D, inverting is trivial.
    if d == 1:
        return n

    # 2D, recursion base case.
    else:  # d == 2:
        w = math.floor((math.sqrt(8 * n[0] + 1) - 1) / 2)
        t = (w ** 2 + w) / 2
        y = int(n[0] - t)
        x = int(w - y)
        return inverse_cantor_pairing([x, y] + n[1:], d=d-1)
