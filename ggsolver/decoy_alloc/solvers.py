import ggsolver.graph as ggraph
import ggsolver.models as models

from ggsolver.dtptb import SWinReach

import concurrent.futures
import os
from itertools import combinations


class EnumerativeTrapsAllocator(models.Solver):
    def __init__(self, graph: ggraph.Graph, num_decoys, decoy_subsets=None, use_multiprocessing=False):
        self.graph = graph
        self.num_decoys = num_decoys
        self.decoy_subsets = decoy_subsets
        self.use_multiprocessing = use_multiprocessing

    def solve(self):
        # Calculate all combinations of decoys
        if self.decoy_subsets is None:
            decoy_combinations = combinations(self.graph.nodes(), self.num_decoys)
        else:
            arena_points = self.decoy_subsets.keys()
            decoy_combinations = combinations(arena_points, self.num_decoys)
        # Evaluate value of deception for each decoy combination
        if self.use_multiprocessing:
            with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
                if self.decoy_subsets is None:
                    futures = [executor.submit(get_value_of_deception_pair, self.graph, decoy_combination)
                               for decoy_combination in decoy_combinations]
                else:
                    futures = [executor.submit(get_value_of_deception_pair, self.graph, decoy_combination,
                                               self.decoy_subsets) for decoy_combination in decoy_combinations]
                # Wait for all value of deception pairs to be calculated
                completed_futures, _ = concurrent.futures.wait(futures)
                decoy_winning_regions = [future.result() for future in completed_futures]
                # Return the decoy combination with the highest value of deception
                highest_value_decoy_set = max(decoy_winning_regions,
                                              key=lambda decoy_set: len(decoy_set["value_of_deception"]))
                return highest_value_decoy_set
        else:
            decoy_value_of_deceptions = list()
            solution_count = 0

            for decoy_combination in decoy_combinations:
                if self.decoy_subsets is None:
                    pair = get_value_of_deception_pair(self.graph, decoy_combination, solution_count=solution_count)
                else:
                    pair = get_value_of_deception_pair(self.graph, decoy_combination, self.decoy_subsets,
                                                       solution_count=solution_count)
                decoy_value_of_deceptions.append(pair)
                solution_count += 1

            highest_value_decoy_set = max(decoy_value_of_deceptions,
                                          key=lambda decoy_set: len(decoy_set["value_of_deception"]))
            return highest_value_decoy_set



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


def get_value_of_deception_pair(graph, decoy_combination, cfg_dict: dict, decoy_subsets=None, metric="winning_states",
                                solution_count=0):
    """ Returns the (decoy,vod) pair for a given decoy combination"""
    final_states = set()
    if decoy_subsets is not None:
        for decoy in decoy_combination:
            final_states = final_states + decoy_subsets[decoy]
    else:
        for decoy in decoy_combination:
            final_states.add(decoy)
    # Create sub graph with final states as sink states
    out_going_final_edges = [graph.out_edges(state) for state in final_states]
    sink_graph = ggraph.SubGraph(graph)
    sink_graph.hide_edges(out_going_final_edges)
    # Solve new game
    solver = SWinReach(sink_graph, final=final_states)
    solver.solve()
    solver.solution().save(os.path.join(cfg_dict['directory'],
                                        f"{cfg_dict['name']}_{solution_count}.solution"), overwrite=True)

    if metric == "winning_states":
        pair = {"decoys": decoy_combination, "value_of_deception": solver.winning_states(1)}
        return pair
    else:
        raise NotImplementedError
