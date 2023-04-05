import multiprocessing

import ggsolver.graph as ggraph
import ggsolver.models as models
import concurrent.futures
import os
import math
from itertools import combinations
from ggsolver.dtptb.pgsolver import SWinReach
import loguru

logger = loguru.logger

MAX_COMBINATIONS = 100


class EnumerativeTrapsAllocator(models.Solver):
    def __init__(self, graph: ggraph.Graph,
                 num_decoys: int,
                 max_combinations=MAX_COMBINATIONS,
                 cpu_count=0,
                 directory=None,
                 fname=None
                 ):
        super(EnumerativeTrapsAllocator, self).__init__(graph)
        self.num_decoys = num_decoys
        self.max_combinations = max_combinations
        self.cpu_count = multiprocessing.cpu_count() if cpu_count == "all" else cpu_count
        self.directory = directory
        self.fname = fname

        self._value_of_deception = self._solution["value_of_deception"] = dict()

    def _multicore_solve(self, decoy_combinations):
        # with concurrent.futures.ProcessPoolExecutor(max_workers=self.cpu_count) as executor:
        #     args = (
        #         (self.graph(), [self._graph["state"][uid] for uid in decoys],
        #          i, "winning_states", self.directory, self.fname)
        #         for i, decoys in enumerate(decoy_combinations)
        #     )
        #     results = executor.map(get_value_of_deception_pair, args)
        #
        #     for result in results:
        #         print(result)
        #
        #     return max(result, key=lambda decoy_set: len(decoy_set["value_of_deception"]))
        raise NotImplementedError("Multicore is not supported due to pickling issues with SubGraph class.")

    def _singlecore_solve(self, decoy_combinations):
        results = []
        for i, decoys in enumerate(decoy_combinations):
            decoys = [self._graph["state"][uid] for uid in decoys]
            args = (self.graph(), decoys, i, "winning_states", self.directory, self.fname)
            result = get_value_of_deception_pair(args)
            results.append(result)
            logger.debug(f"Solved deceptive planning for {decoys=}.")

        return max(results, key=lambda decoy_set: len(decoy_set["value_of_deception"]))

    def solve(self):
        """
        # FIXME: Not checking node siblings for now. (decoy_subsets/arena_maping or so.)
        :return:
        """
        # Check for computability
        num_combinations = math.comb(self._graph.number_of_nodes(), self.num_decoys)
        if num_combinations > self.max_combinations:
            raise RuntimeError(f"Cannot process more than {self.max_combinations} games.")
        logger.debug(f"Setting up solvers for {num_combinations} games.")

        # Define combinations and extract the final states.
        decoy_combinations = combinations(self._graph.nodes(), self.num_decoys)

        # Based on multiprocessing, solve for each decoy placement.
        if self.cpu_count > 1:
            self._multicore_solve(decoy_combinations)
        else:
            self._singlecore_solve(decoy_combinations)

        self._is_solved = True


class GreedyTrapsAllocator(models.Solver):
    def __init__(self, graph: ggraph.Graph, num_decoys, use_multiprocessing=False):
        pass

    def solve(self):
        pass


class EnumerativeFakesAllocator(models.Solver):
    def __init__(self, graph: ggraph.Graph, num_decoys, use_multiprocessing=False):
        pass

    def solve(self):
        pass


class GreedyFakesAllocator(models.Solver):
    def __init__(self, graph: ggraph.Graph, num_decoys, use_multiprocessing=False):
        pass

    def solve(self):
        pass


class EnumerativeMixedAllocator(models.Solver):
    def __init__(self, graph: ggraph.Graph, num_decoys, use_multiprocessing=False):
        pass

    def solve(self):
        pass


class GreedyMixedAllocator(models.Solver):
    def __init__(self, graph: ggraph.Graph, num_decoys, use_multiprocessing=False):
        pass

    def solve(self):
        pass


def get_value_of_deception_pair(args):
    """ Returns the (decoy,vod) pair for a given decoy combination"""
    logger.debug(f"{args}")
    graph, decoys, solution_count, metric, directory, f_name = args

    # Solve new game
    solver = SWinReach(graph, final=decoys)
    solver.solve()
    logger.info(f"Solved game {f_name}_{solution_count} with {decoys}.")

    if directory is not None and f_name is not None:
        solver.solution().save(os.path.join(directory, f"{f_name}_{solution_count}.solution"))

    if metric == "winning_states":
        pair = {"decoys": decoys, "value_of_deception": solver.winning_states(1), "solver": solver}
        return pair
    else:
        raise NotImplementedError
