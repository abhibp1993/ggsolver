from ggsolver.dtptb import SWinReach
import ggsolver.graph as gg_graph
from itertools import combinations
import os
import datetime
import concurrent.futures

CONFIG = {
    "directory": "out",
    "filename": f"{datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}",
    "solution_counter": 0,
}

def calc_win_region_sink_states(graph, final_states):
    """ Returns the winning region of a given graph with the final states as sink states"""
    out_going_final_edges = [graph.out_edges(state) for state in final_states]
    sink_graph = gg_graph.SubGraph(graph)
    sink_graph.hide_edges(out_going_final_edges)

    solver = SWinReach(sink_graph, final=final_states)
    solver.solve()
    solver.solution().save(os.path.join(CONFIG["directory"],
                                        f"{CONFIG['filename']}_{CONFIG['solution_counter']}.solution"), overwrite=True)
    CONFIG["solution_counter"] += 1

    # TODO add different metrics to determine value of deception
    return solver.winning_states(1)

def exhaustive_search_subsets(graph, decoy_subsets, max_decoys=int("inf")):
    decoy_winning_regions = list()
    arena_points = decoy_subsets.keys()

    decoy_combinations = combinations(arena_points, max_decoys)

    for decoy_combination in decoy_combinations:
        final_states = set()
        for decoy in decoy_combination:
            final_states = final_states + decoy_subsets[decoy]

        pair = {"decoys": decoy_combination, "value_of_deception": calc_win_region_sink_states(graph, final_states)}
        decoy_winning_regions.append(pair)

    highest_value_decoy_set = max(decoy_winning_regions, key=lambda decoy_set: len(decoy_set["value_of_deception"]))
    return highest_value_decoy_set

def exhaustive_search(graph, max_decoys=int("inf")):
    decoy_value_of_deceptions = list()

    decoy_combinations = combinations(graph.nodes(), max_decoys)
    for decoy_combination in decoy_combinations:
        final_states = set()
        for decoy in decoy_combination:
            set.add(decoy)
        pair = {"decoys": decoy_combination, "value_of_deception": calc_win_region_sink_states(graph, final_states)}
        decoy_value_of_deceptions.append(pair)

    highest_value_decoy_set = max(decoy_value_of_deceptions, key=lambda decoy_set: len(decoy_set["value_of_deception"]))
    return highest_value_decoy_set
def solve_trap_exhaustive(game, max_decoys=float("inf"), decoy_subsets=None):
    """
    decoy_subsets is a dict mapping arena points to the states that become decoys if that arena point is a decoy
    """
    base_graph = game.graphify()
    base_graph.save(os.path.join(CONFIG["directory"], f"{CONFIG['filename']}.ggraph"), overwrite=True)

    if decoy_subsets is not None:
        return exhaustive_search_subsets(base_graph, decoy_subsets, max_decoys)
    else:
        return exhaustive_search(base_graph, max_decoys)