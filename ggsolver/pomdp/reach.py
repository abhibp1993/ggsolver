import random
# from ggsolver.models import Solver
import ggsolver.models as models
from tqdm import tqdm


class ASWinReach(models.Solver):
    def __init__(self, graph, final=None, player=1, **kwargs):
        """
        Instantiates a sure winning reachability game solver.

        :param graph: (Graph instance)
        :param final: (iterable) A list/tuple/set of final nodes in graph.
        :param player: (int) Either 1 or 3.
        :param kwargs: SureWinReach accepts no keyword arguments.
        """
        super(ASWinReach, self).__init__(graph, **kwargs)
        self._player = player
        self._final = set(final) if final is not None else {n for n in graph.nodes() if self._graph["final"][n] == 0}

    def solve(self):
        """
        Custom algorithm of Sumukha!
        """
        # TODO. self._solution -> SubGraph.

        # Mark the game to be solved
        self._is_solved = True

    def progress(self, set_r, set_y):
        pass

    def allow(self, uid, set_y):
        pass

    def allow_equivalent(self, uid, set_y):
        pass
