import itertools
import random

import ggsolver.graph as ggraph
import ggsolver.dtptb as dtptb


class Mesh(dtptb.DTPTBGame):
    def __init__(self, num_nodes):
        super(Mesh, self).__init__()
        self.num_nodes = num_nodes
        self.num_final = num_nodes // 10     # Arbitrary choice
        self._turn_cache = dict()

    def __str__(self):
        delta = {(st, a): self.delta(st, a) for st in self.states() for a in self.enabled_acts(st)}
        final = {st for st in self.states() if self.final(st)}
        return f"{self.states()=}\n" \
               f"{delta=}\n" \
               f"{final=}\n"

    def states(self):
        return [f"s{i}" for i in range(self.num_nodes)]

    def turn(self, state):
        if state not in self._turn_cache:
            turn = random.choice([1, 2])
            self._turn_cache[state] = turn
            return turn

    def actions(self):
        return list()

    def enabled_acts(self, state):
        i = int(state[1:])
        return [str((i, j)) for j in range(self.num_nodes) if j != i]

    def delta(self, state, act):
        i, j = eval(act)
        if int(state[1:]) == i:
            return f"s{j}"

    def final(self, state):
        i = int(state[1:])
        if i < self.num_nodes // 10:
            return True
        return False


def mesh(config):
    return Mesh(num_nodes=config['graph']['nodes'])


def ring(config):
    return None


def star(config):
    return None


def tree(config):
    return None


def hybrid(config):
    return None
