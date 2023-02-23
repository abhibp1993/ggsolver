"""
EEL 4930/5934 Formal Methods in Robotics and AI
Instructor: Dr. Jie Fu
TAs: Abhishek N. Kulkarni and Haoxiang Ma

HW3: Planning in Deterministic Transition Systems using LTL objectives
Task:
    In this assignment, you will implement a planner for Jerry in a deterministic gridworld.
    See HW3.pdf for problem description.

Reference:
    1. API Docs: https://akulkarni.me/ggsolver/modules/models.html#ggsolver.models.TSys
    2. Example: https://akulkarni.me/ggsolver/examples/models_tsys.html

This example is used to reproduce and fix the bug on line 209.
"""
import itertools
from ggsolver.models import Game, TSys
from ggsolver.logic import SpotAutomaton


class TomAndJerry(Game):
    def __init__(self, dim, traps, cheese, home):
        """
        Parameters for the gridworld.

        :param dim: Dimensions of gridworld (row, col)
        :param batt: Maximum battery level.
        :param traps: List of cells (r, c) that contain an obstacle.
        :param cheese: List of cells (r, c) that contain a flag.
        """
        assert dim[0] > 0 and dim[1] > 0
        super(TomAndJerry, self).__init__(is_deterministic=True)
        self.dim = dim
        self.traps = traps
        self.cheese = cheese
        self.home = home

    def states(self):
        """
        Returns a list of states in gridworld.
        A state is represented as (tom.row, tom.col, tom.dir, jerry.row, jerry.col).
        """
        rows = self.dim[0]
        cols = self.dim[1]
        return [
            (tr, tc, tdir, jr, jc)
            for tr, tc, tdir, jr, jc in itertools.product(range(rows), [3], ["N", "S"], range(rows), range(cols))
        ]

    def actions(self):
        """
        Return a list of actions. Each action is identified by a string label.

        In each round, both Tom and Jerry must select an action from ["N", "E", "S", "W"].
        Thus, each action is represented as a tuple of (tom.action, jerry.action).
        """
        return ["N", "E", "S", "W"]

    def delta(self, state, inp):
        """
        Implement the transition function.

        :param state: A state from the list returned by states().
        :param inp: An action from the list returned by actions().
        :return: The next state, which is the result of applying the action `inp` to `state`.
        """
        tr, tc, tdir, jr, jc = state

        # If jerry is caught or trapped, game end (no state change).
        if (jr, jc) in self.traps or (tr, tc) == (jr, jc):
            return tr, tc, tdir, jr, jc

        # Otherwise, move Tom.
        next_tr, next_tc = self.apply_action((tr, tc), tdir)
        if (next_tr, next_tc) == (tr, tc):
            next_tdir = "N" if tdir == "S" else "S"
            next_tr, next_tc = self.apply_action((tr, tc), next_tdir)
        else:
            next_tdir = tdir

        # Move Jerry
        next_jr, next_jc = self.apply_action((jr, jc), inp)

        return next_tr, next_tc, next_tdir, next_jr, next_jc

    def atoms(self):
        """
        Returns a list of atomic propositions. Each atomic proposition is a string.
        """
        return ["caught", "cheese", "trapped", "home"]

    def label(self, state):
        """
        Returns a list of atoms that are true in the `state`.
        :param state: A state from the list returned by states().
        :return: List of atoms.
        """
        tr, tc, tdir, jr, jc = state
        if (jr, jc) in self.traps:
            return ["trapped"]          # Jerry is trapped in traps set by Tom

        if (jr, jc) == (tr, tc):
            return ["caught"]           # Jerry is caught by Tom

        if (jr, jc) in self.cheese:
            return ["cheese"]           # Jerry is at a cheese cell

        if (jr, jc) in self.home:
            return ["home"]             # Jerry is at home cell

        return list()

    def apply_action(self, cell, act):
        row, col = cell

        if act == "N":
            return (row + 1, col) if 0 <= row + 1 < self.dim[0] else (row, col)
        elif act == "E":
            return (row, col + 1) if 0 <= col + 1 < self.dim[1] else (row, col)
        elif act == "S":
            return (row - 1, col) if 0 <= row - 1 < self.dim[0] else (row, col)
        else:  # inp == "W":
            return (row, col - 1) if 0 <= col - 1 < self.dim[1] else (row, col)


class ProductTSysDFA(Game):
    def __init__(self, tsys, dfa):
        super(ProductTSysDFA, self).__init__(is_deterministic=True)
        self.tsys = tsys
        self.dfa = dfa

    def states(self):
        return list(itertools.product(self.tsys.states(), self.dfa.states()))

    def actions(self):
        return self.tsys.actions()

    def delta(self, state, act):
        s, q = state
        t = self.tsys.delta(s, act)
        # label = list(set(self.tsys.label(t)) - {"trapped", "caught"})
        label = list(set(self.tsys.label(t)))   # - {"trapped", "caught"})
        p = self.dfa.delta(q, label)
        return t, p

    def init_state(self):
        s0 = self.tsys.init_state()
        q0 = self.dfa.init_state()
        return s0, self.dfa.delta(q0, self.tsys.label(s0))

    def final(self, state):
        return True if self.dfa.final(state[1]) != [] else False
        # final = [st for st in self.dfa.states() if self.dfa.final(st) == 0]
        # return list(itertools.product(self.tsys.states(), final))


class ReachabilityGameSolver:
    def __init__(self, game):
        self.game = game
        self.attr = None
        self.pi = None

    def reachability(self):
        """
        Compute the states from which jerry can avoid being "caught".

        :param game: TomAndJerry instance.
        :return:
        """
        # TODO 1. Identify the states that Jerry must remain within.
        states = set(self.game.states())
        final = {st for st in states if self.game.final(st)}

        # TODO 2. Implement invariance computation algorithm from slide 22.
        level_sets = [final]
        while True:
            attr = {st for st in states - level_sets[-1]
                    if any(self.game.delta(st, act) in level_sets[-1] for act in self.game.actions())}
            level_sets.append(set.union(level_sets[-1], attr))
            if level_sets[-1] == level_sets[-2]:
                break

        # TODO 3. Define rank of each state
        rank = {st: float("inf") for st in states}
        for st in states:
            for idx in range(len(level_sets)):
                if st in level_sets[idx]:
                    rank[st] = idx
                    break

        # TODO 4. Construct strategy
        pi = {st: [] for st in states}
        for st in states:
            for act in self.game.actions():
                if rank[self.game.delta(st, act)] < rank[st]:
                    pi[st].append(act)

        # print(recur, pi)
        return level_sets, pi

    def solve(self):
        self.attr, self.pi = self.reachability()
        return self.attr, self.pi


def gen_automaton():
    # aut = SpotAutomaton("F(cheese & F home)")
    aut = SpotAutomaton("F(cheese & F home)", atoms=["cheese", "caught", "trapped", "home"])
    return aut


def main_reachability(tsys, aut):
    prod = ProductTSysDFA(tsys, aut)
    solver = ReachabilityGameSolver(prod)
    return solver.solve()


if __name__ == '__main__':
    tsys = TomAndJerry(dim=(4, 4), traps=[(0, 0), (1, 1)], home=(0, 1), cheese=[(2, 2), (0, 3)])
    aut = gen_automaton()

    attr, pi = main_reachability(tsys, aut)
    print(len(attr))

