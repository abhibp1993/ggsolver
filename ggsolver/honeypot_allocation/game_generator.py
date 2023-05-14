import itertools
import random
import ggsolver.graph as ggraph
import ggsolver.dtptb as dtptb

random.seed(42)


class Mesh(dtptb.DTPTBGame):
    def __init__(self, num_nodes):
        super(Mesh, self).__init__()
        self.num_nodes = num_nodes
        self.num_final = num_nodes // 10  # Arbitrary choice
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


class Hybrid(dtptb.DTPTBGame):
    def __init__(self, num_nodes, max_out_degree):
        assert isinstance(max_out_degree, int), f"{max_out_degree=}. Expected an integer > 0."
        super(Hybrid, self).__init__()
        self.num_nodes = num_nodes
        self.max_out_degree = max_out_degree
        self.num_final = num_nodes // 10  # Arbitrary choice
        self._turn_cache = dict()
        self._trans_dict = dict()

        for i in range(self.num_nodes):
            num_successors = random.randint(1, self.max_out_degree)
            successors = (random.choices(range(self.num_nodes), k=num_successors))
            self._trans_dict[f"s{i}"] = {(f"s{i}", f"s{j}"): f"s{j}" for j in successors}

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
        return list(self._trans_dict[state].keys())

    def delta(self, state, act):
        return self._trans_dict.get(state, dict()).get(act, None)

    def final(self, state):
        i = int(state[1:])
        if i < self.num_nodes // 10:
            return True
        return False


def mesh(config):
    return Mesh(num_nodes=config['graph']['nodes'])


def hybrid(config):
    return Hybrid(num_nodes=config['graph']['nodes'], max_out_degree=config['graph']['max_out_degree'])
