import copy
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
