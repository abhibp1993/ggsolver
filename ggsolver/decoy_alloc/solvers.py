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

MAX_COMBINATIONS = 100000


class EnumerativeTrapsAllocator(models.Solver):
    """
    :param graph: graph of hypergame
    """

    def __init__(self, graph: ggraph.Graph,
                 num_decoys: int,
                 max_combinations=MAX_COMBINATIONS,
                 cpu_count=0,
                 directory=None,
                 fname=None,
                 save_output=False
                 ):
        super(EnumerativeTrapsAllocator, self).__init__(graph)
        self.num_decoys = num_decoys
        self.max_combinations = max_combinations
        self.cpu_count = multiprocessing.cpu_count() if cpu_count == "all" else cpu_count
        self.directory = directory
        self.fname = fname
        self._save_output = save_output

        # dict with the decoys, VOD, and solver for the best decoy allocation
        self.deception_dict = None

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
            # Remove out going edges from decoy states
            # (the hypergame only has out going edges removed from the original final states)
            sub_graph = remove_out_going_final_edges(self.graph(), decoys, self._state2node)
            # Solve the sub_graph
            args = (sub_graph, decoys, i, "winning_states", self.directory, self.fname, self._save_output)
            result = get_value_of_deception_pair(args)
            results.append(result)
            self._value_of_deception[str(result["decoys"])] = result["value_of_deception"]
            logger.debug(f"Solved deceptive planning for {decoys=}.")

        return max(results, key=lambda decoy_set: decoy_set["value_of_deception"])

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
        possible_decoys = set(self._graph.nodes()) - set(
            uid for uid in self.graph().nodes() if self.graph()["final"][uid])
        decoy_combinations = combinations(possible_decoys, self.num_decoys)

        # Solve trivial case where number of potential decoy states is less than number decoys to allocate
        if len(possible_decoys) < self.num_decoys:
            possible_decoy_states = [self._graph["state"][uid] for uid in possible_decoys]
            sub_graph = remove_out_going_final_edges(self.graph(), possible_decoy_states, self._state2node)
            args = (sub_graph, possible_decoy_states, 0, "winning_states", self.directory, self.fname, self._save_output)
            self.deception_dict = get_value_of_deception_pair(args)
        # Based on multiprocessing, solve for each decoy placement.
        elif self.cpu_count > 1:
            self.deception_dict = self._multicore_solve(decoy_combinations)
        else:
            self.deception_dict = self._singlecore_solve(decoy_combinations)

        self._edge_winner.update(self.deception_dict["solver"]._edge_winner)
        self._node_winner.update(self.deception_dict["solver"]._node_winner)
        self._solution["vod"] = self.deception_dict["value_of_deception"]
        self._is_solved = True


class GreedyTrapsAllocator(models.Solver):
    def __init__(self, graph: ggraph.Graph,
                 num_decoys: int,
                 max_combinations=MAX_COMBINATIONS,
                 cpu_count=0,
                 directory=None,
                 fname=None,
                 save_output=False
                 ):
        super(GreedyTrapsAllocator, self).__init__(graph)
        self.num_decoys = num_decoys
        self.max_combinations = max_combinations
        self.cpu_count = multiprocessing.cpu_count() if cpu_count == "all" else cpu_count
        self.directory = directory
        self.fname = fname
        self._save_output = save_output

        # dict with the decoys, VOD, and solver for the best decoy allocation
        self.deception_dict = None

        self._value_of_deception = self._solution["value_of_deception"] = dict()

    def _multicore_solve(self):
        raise NotImplementedError("Multicore is not supported due to pickling issues with SubGraph class.")

    def _singlecore_solve(self):
        states = set(self._graph["state"][uid] for uid in self._graph.nodes())
        final_states = set(self.graph()["state"][uid] for uid in self.graph().nodes() if self.graph()["final"][uid])
        trap_states = set()
        covered_states = set()
        iter_count = 0
        while len(states - covered_states) > 0 and len(trap_states) < self.num_decoys:
            iter_count += 1
            potential_traps = states - trap_states - final_states
            updated_winning_regions = list()
            # Consider the trivial case with more available decoys than states to potentially make traps
            if len(potential_traps) <= self.num_decoys:
                sub_graph = remove_out_going_final_edges(self.graph(), potential_traps, self._state2node)
                args = (sub_graph, potential_traps, iter_count, "winning_states", self.directory, self.fname,
                        self._save_output)
                return get_value_of_deception_pair(args)

            for potential_trap in potential_traps:
                new_final_states = trap_states.union(set([potential_trap]))
                # Remove out going edges from decoy states
                # (the hypergame only has out going edges removed from the original final states)
                sub_graph = remove_out_going_final_edges(self.graph(), new_final_states, self._state2node)
                # Solve the sub_graph
                args = (sub_graph, new_final_states, iter_count, "winning_states", self.directory, self.fname,
                        self._save_output)
                result = get_value_of_deception_pair(args)
                new_region = {"result": result, "new_trap": potential_trap}
                updated_winning_regions.append(new_region)
            next_trap_set = max(updated_winning_regions,
                                key=lambda decoy_set: decoy_set["result"]["value_of_deception"])
            trap_states.add(next_trap_set["new_trap"])
            covered_states.update(next_trap_set["result"]["solver"].winning_states(1))
        return next_trap_set["result"]

    def solve(self):
        # Based on multiprocessing, solve for each decoy placement.
        if self.cpu_count > 1:
            self.deception_dict = self._multicore_solve()
        else:
            self.deception_dict = self._singlecore_solve()

        self._edge_winner.update(self.deception_dict["solver"]._edge_winner)
        self._node_winner.update(self.deception_dict["solver"]._node_winner)
        self._solution["vod"] = self.deception_dict["value_of_deception"]
        self._is_solved = True


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


def remove_out_going_final_edges(graph: ggraph.Graph, final_states: set, state2node):
    hidden_edges = set()
    out_going_final_edges = list()
    for state in final_states:
        for out_edge in graph.out_edges(state2node[state]):
            out_going_final_edges.append(out_edge)
    hidden_edges.update(out_going_final_edges)
    sub_graph = ggraph.SubGraph(graph, hidden_edges=hidden_edges)
    return sub_graph


# TODO (Note.). The following function is only for traps.
def get_value_of_deception_pair(args):
    """ Returns the (decoy,vod) pair for a given decoy combination"""
    logger.debug(f"{args}")
    graph, decoys, solution_count, metric, directory, f_name, save_output = args

    # Solve new game
    solver = SWinReach(graph, final=decoys, path=directory, save_output=save_output,
                       filename=f"pgzlk_{'_'.join(decoys)}")
    solver.solve()
    logger.info(f"Solved game {f_name}_{solution_count} with {decoys}.")

    if directory is not None and f_name is not None:
        solver.solution().save(os.path.join(directory, f"{f_name}_{solution_count}.solution"), overwrite=True)

    if metric == "winning_states":
        # FIXME Define value of deception based on paper instead of number of winning states
        vod = len(solver.winning_states(1)) / \
              (graph.number_of_nodes() - len([uid for uid in graph.nodes() if graph["final"][uid]]))
        pair = {"decoys": decoys, "value_of_deception": vod, "solver": solver}
        return pair
    else:
        raise NotImplementedError
