import itertools
import ggsolver.logic.automata as automata
from functools import reduce


class DFACrossProduct(automata.DFA):
    def __init__(self, automata):
        super(DFACrossProduct, self).__init__()
        self.automata = list(automata)
        assert len(self.automata) > 0, "There should be at least one DFA to compute product!"

    def states(self):
        return list(itertools.product(*[dfa.states() for dfa in self.automata]))

    def atoms(self):
        return list(reduce(set.union, [set(dfa.atoms()) for dfa in self.automata]))

    def init_state(self):
        return tuple(dfa.init_state() for dfa in self.automata)

    def delta(self, state, inp):
        return tuple(self.automata[i].delta(state[i], inp) for i in range(len(self.automata)))


class DFAIntersectionProduct(DFACrossProduct):
    def final(self, state):
        """
        DFAs have single acceptance set. Hence, we assert acceptance set of final states to be 0.
        """
        return all(dfa.final(state[0]) == 0 for dfa in self.automata)


class DFAUnionProduct(DFACrossProduct):
    def final(self, state):
        """
        DFAs have single acceptance set. Hence, we assert acceptance set of final states to be 0.
        """
        return any(dfa.final(state[0]) == 0 for dfa in self.automata)