import itertools


def apply_atoms_limit(atoms):
    if len(atoms) > 16:
        raise ValueError("ggsolver supports atoms set up to size 16.")


def powerset(iterable):
    s = list(iterable)
    return itertools.chain.from_iterable(itertools.combinations(s, r) for r in range(len(s) + 1))
