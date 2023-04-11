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


def get_value_of_deception_pair(graph, decoy_combination, decoy_subsets=None):
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
    sink_graph = gg_graph.SubGraph(graph)
    sink_graph.hide_edges(out_going_final_edges)
    # Solve new game
    solver = SWinReach(sink_graph, final=final_states)
    solver.solve()
    solver.solution().save(os.path.join(CONFIG["directory"],
                                        f"{CONFIG['filename']}_{CONFIG['solution_counter']}.solution"), overwrite=True)
    CONFIG["solution_counter"] += 1

    # TODO add different metrics to determine value of deception
    pair = {"decoys": decoy_combination, "value_of_deception": solver.winning_states(1)}
    return pair


def exhaustive_search_subsets(graph, decoy_subsets, max_decoys=float("inf")):
    arena_points = decoy_subsets.keys()
    decoy_combinations = combinations(arena_points, max_decoys)

    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(get_value_of_deception_pair, graph, decoy_combination, decoy_subsets)
                   for decoy_combination in decoy_combinations]
        # Wait for all value of deception pairs to be calculated
        completed_futures, _ = concurrent.futures.wait(futures)
        decoy_winning_regions = [future.result() for future in completed_futures]
        # Return the decoy combination with the highest value of deception
        highest_value_decoy_set = max(decoy_winning_regions, key=lambda decoy_set: len(decoy_set["value_of_deception"]))
        return highest_value_decoy_set


def exhaustive_search(graph, max_decoys=float("inf")):
    decoy_combinations = combinations(graph.nodes(), max_decoys)
    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(get_value_of_deception_pair, graph, decoy_combination)
                   for decoy_combination in decoy_combinations]
        # Wait for all value of deception pairs to be calculated
        completed_futures, _ = concurrent.futures.wait(futures)
        decoy_winning_regions = [future.result() for future in completed_futures]
        # Return the decoy combination with the highest value of deception
        highest_value_decoy_set = max(decoy_winning_regions, key=lambda decoy_set: len(decoy_set["value_of_deception"]))
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


if __name__ == '__main__':
    g_graph = gg_graph.Graph()
    g_graph.add_nodes(100)
    for node in g_graph.nodes():
        for node2 in g_graph.nodes():
            g_graph.add_edge(node, node2)
    g_graph.save(os.path.join(CONFIG["directory"], "base_game.ggraph"))
    # decoys = exhaustive_search(g_graph, 2)
    # print(decoys)
