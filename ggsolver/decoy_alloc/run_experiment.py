"""
Runs experiment and generates report based on cfg_dicturation files.
"""
import os.path
import cProfile
import random
import time

import networkx as nx
import matplotlib.pyplot as plt
import pygraphviz
from pygraphviz import graphviz

import ggsolver.decoy_alloc.process_config as cfg
import ggsolver.decoy_alloc.graph_generator as gen
import ggsolver.decoy_alloc.solvers as solvers
import ggsolver.decoy_alloc.models as models
from ggsolver.dtptb.pgsolver import SWinReach

from examples.apps.tom_and_jerry.main import TomJerryGame

import ggsolver.graph as ggraph
import ggsolver.models as gmodels
import ggsolver.dtptb as dtptb

import loguru

# from memory_profiler import profile

logger = loguru.logger
logger.remove()

# CONFIG_FILE_PATH = "configurations/config1.json"
CONFIG_FILE_PATH = "configurations/enumerative_exp_n10_hybrid_t3_f0.json"


def write_dot_file(graph: ggraph.Graph, game_name, cfg_dict: dict, **kwargs):
    path = os.path.join(cfg_dict['directory'], f"{cfg_dict['name']}_{game_name}.dot")
    with open(path, 'w') as file:
        contents = list()
        contents.append("digraph G {\n")

        for node in graph.nodes():
            node_properties = {
                "shape": 'circle' if graph['turn'][node] == 1 else 'box',
                "label": graph['state'][node],
                "peripheries": '2' if graph['final'][node] else '1',
            }
            if "node_winner" in graph.node_properties:
                node_properties |= {"color": 'blue' if graph['node_winner'][node] == 1 else 'red'}

            contents.append(
                f"N{node} [" + ", ".join(f'{k}="{v}"' for k, v in node_properties.items()) + "];\n"
            )

            for uid, vid, key in graph.out_edges(node):
                edge_properties = {
                    "label": graph["input"][uid, vid, key] if kwargs.get("no_actions", False) else ""
                }
                if "edge_winner" in graph.edge_properties:
                    edge_properties |= {"color": 'blue' if graph['edge_winner'][uid, vid, key] == 1 else 'red'}

                contents.append(
                    f"N{uid} -> N{vid} [" + ", ".join(f'{k}="{v}"' for k, v in edge_properties.items()) + "];\n"
                )

        contents.append("}")
        file.writelines(contents)

    # Generate SVG
    g = pygraphviz.AGraph(path)
    g.layout('dot')
    path = os.path.join(cfg_dict['directory'], f"{cfg_dict['name']}_{game_name}.svg")
    g.draw(path=path, format='svg')


def gen_game(cfg_dict: dict) -> dtptb.DTPTBGame:
    topology = cfg_dict['graph']['topology']
    if topology == "mesh":
        return gen.mesh(cfg_dict)
    else:  # type == "hybrid":
        return gen.hybrid(cfg_dict)


# @profile
def place_decoys(graph: ggraph.Graph, cfg_dict: dict, arena2state: dict = None) -> gmodels.Solver:
    type_ = cfg_dict['type']
    num_traps = cfg_dict['max_traps']
    num_fakes = cfg_dict['max_fakes']
    cpu_count = cfg_dict['use_multiprocessing']

    if type_ == "enumerative" and num_traps > 0 and num_fakes == 0:
        logger.info("Running EnumerativeTrapsAllocator...")
        solution = solvers.EnumerativeTrapsAllocator(graph=graph,
                                                     num_decoys=num_traps,
                                                     cpu_count=cpu_count,
                                                     directory=cfg_dict['directory'],
                                                     fname=cfg_dict['name'],
                                                     save_output=cfg_dict["save_intermediate_solutions"]
                                                     )

    elif type_ == "greedy" and num_traps > 0 and num_fakes == 0:
        solution = solvers.GreedyTrapsAllocator(graph=graph,
                                                arena2states=arena2state,
                                                num_decoys=num_traps,
                                                cpu_count=cpu_count,
                                                directory=cfg_dict['directory'],
                                                fname=cfg_dict['name'],
                                                save_output=cfg_dict["save_intermediate_solutions"]
                                                )

    elif type_ == "enumerative" and num_traps == 0 and num_fakes > 0:
        solution = solvers.EnumerativeFakesAllocator(graph=graph,
                                                     num_decoys=num_fakes,
                                                     use_multiprocessing=cpu_count)

    elif type_ == "greedy" and num_traps == 0 and num_fakes > 0:
        solution = solvers.GreedyFakesAllocator(graph=graph,
                                                num_decoys=num_fakes,
                                                use_multiprocessing=cpu_count)

    elif type_ == "enumerative" and num_traps > 0 and num_fakes > 0:
        solution = solvers.EnumerativeMixedAllocator(graph=graph,
                                                     num_decoys=(num_traps, num_fakes),
                                                     use_multiprocessing=cpu_count
                                                     )

    else:  # type_ == "greedy" and num_traps == 0 and num_fakes > 0:
        solution = solvers.GreedyMixedAllocator(graph=graph,
                                                num_decoys=(num_traps, num_fakes),
                                                use_multiprocessing=cpu_count)

    # Solve the game
    solution.solve()
    return solution


def place_traps_and_fakes_greedy(graph: ggraph.Graph, cfg_dict: dict,
                                 arena2state: dict = None) -> gmodels.Solver:
    num_traps = cfg_dict['max_traps']
    num_fakes = cfg_dict['max_fakes']
    cpu_count = cfg_dict['use_multiprocessing']

    # Allocate fakes
    fakes_solution = solvers.GreedyFakesAllocator(graph=graph,
                                                  arena2states=arena2state,
                                                  num_fakes=num_fakes,
                                                  cpu_count=cpu_count,
                                                  directory=cfg_dict['directory'],
                                                  fname=cfg_dict['name'],
                                                  save_output=cfg_dict["save_intermediate_solutions"]
                                                  )
    fakes_solution.solve()
    logger.info(f"Selected fakes {fakes_solution.set_of_fakes=}")
    hypergame = fakes_solution.hypergame
    write_dot_file(hypergame, "hgame", cfg_dict)
    if num_traps == 0:
        return fakes_solution
    # Use the hypergame from fake allocation to allocate traps
    traps_solution = solvers.GreedyTrapsAllocator(graph=hypergame,
                                                  arena2states=arena2state,
                                                  num_decoys=num_traps,
                                                  cpu_count=cpu_count,
                                                  directory=cfg_dict['directory'],
                                                  fname=cfg_dict['name'],
                                                  save_output=cfg_dict["save_intermediate_solutions"]
                                                  )
    traps_solution.solve()
    logger.info(f"Selected traps {traps_solution.deception_dict['decoys']}")
    return traps_solution


def gen_reports(config):
    """
    Use files generated during solution to generate reports.
    :param config: (dict) Configuration dictionary.
    """
    # 1. Generate colored graphs for all intermediate solutions, if option enabled.
    # 2. Generate colored graphs for final allocation.
    # 3.
    pass


def gen_hypergame(game_graph, swin_game: dtptb.SWinReach):
    """
    :param game_graph: base game graph
    :param swin_game: solution to base game
    :param config:
    :return:
    """
    # FIXME. Depending on whether we are allocating only traps, only fakes or both, generate the hypergame.
    hidden_nodes = {uid for uid in swin_game.winning_nodes(1)}
    hidden_edges = set()
    for uid in swin_game.winning_nodes(2):
        for _, vid, key in swin_game.solution().out_edges(uid):
            if swin_game.solution()["rank"][vid] >= swin_game.solution()["rank"][uid]:
                hidden_edges.add((uid, vid, key))

    # Remove outgoing edges from final states
    out_going_final_edges = [game_graph.out_edges(state) for state in swin_game.get_final_states()]
    hidden_edges.update(out_going_final_edges[0])
    hgame_graph = ggraph.SubGraph(game_graph, hidden_nodes=hidden_nodes, hidden_edges=hidden_edges)
    return hgame_graph


def solve_base_game(game_graph: ggraph.Graph, game: dtptb.DTPTBGame, cfg_dict: dict):
    swin_game = SWinReach(game_graph, player=2)
    swin_game.solve()
    path = os.path.join(cfg_dict['directory'], f"{cfg_dict['name']}_base.solution")
    swin_game.solution().save(path)

    logger.info(f"game.final:{[st for st in game.states() if game.final(st)]}.")
    logger.info(f"P1 SWin:{swin_game.winning_states(player=1)}, P2 SWin:{swin_game.winning_states(player=2)}")
    write_dot_file(swin_game.solution(), "base_game", cfg_dict)
    logger.success(f"Solved {game_graph=} successfully.")
    return swin_game


def run_experiment(config):
    # Generate base game
    game = gen_game(config)
    game_graph = game.graphify()
    logger.success(f"Generated {game_graph=} successfully.")
    # Logging and saving graph
    directory = config['directory']
    exp_name = config['name']
    if config['graph']['save']:
        game_graph.save(os.path.join(directory, f'{exp_name}_base.ggraph'))
    if config['graph']['save_png']:
        game_graph.to_png(os.path.join(directory, f'{exp_name}_base.png'), nlabel=["state"], elabel=["input"])
    logger.success(f"Saved {game_graph=} successfully.")

    # Solve base game
    swin_game = solve_base_game(game_graph, game, cfg_dict=config)

    # Construct hypergame graph (Def. 6, in draft as of 4 Apr. 2023)
    hgame_graph = gen_hypergame(game_graph, swin_game)
    path = os.path.join(config['directory'], f"{config['name']}_hgame.ggraph")
    hgame_graph.save(path)
    write_dot_file(hgame_graph, "hgame", config)
    logger.info(f"Constructed and saved {hgame_graph=} successfully.")
    # Check for games where no traps can be placed
    assert len(set(hgame_graph.nodes()) - swin_game.get_final_states()) > 0, \
        f"Hypergame {hgame_graph} has no states where traps can be allocated"
    # assert len(set(hgame_graph.nodes()) - swin_game.get_final_states()) > config["max_traps"], \
    #     f"Hypergame {hgame_graph} has fewer non-final states than traps to be allocated"
    # Allocate decoys
    start = time.perf_counter()
    solution = place_decoys(hgame_graph, config)
    logger.success(f"Decoy placement completed.")
    end = time.perf_counter()
    logger.info(f"Time required to place decoys: {end - start} seconds.")

    # Extract solution graph. Save it, log it.
    sol_graph = solution.solution()
    sol_graph.save(os.path.join(directory, f'{exp_name}_solution.ggraph'))
    write_dot_file(sol_graph, "sol_graph", config)

    if config['graph']['save_png']:
        game_graph.to_png(os.path.join(directory, f'{exp_name}_base.png'),
                          nlabel=["state", "node_winner"],
                          elabel=["input", "edge_winner"])

    # Return variables of interest
    exec_time = end - start
    ram_used = None
    vod = sol_graph["vod"]
    return exec_time, ram_used, vod


def run_mixed_experiment(config):
    # Generate base game
    game = gen_game(config)
    game_graph = game.graphify()
    logger.success(f"Generated {game_graph=} successfully.")
    # Logging and saving graph
    directory = config['directory']
    exp_name = config['name']
    if config['graph']['save']:
        game_graph.save(os.path.join(directory, f'{exp_name}_base.ggraph'))
    if config['graph']['save_png']:
        game_graph.to_png(os.path.join(directory, f'{exp_name}_base.png'), nlabel=["state"], elabel=["input"])
    write_dot_file(game_graph, "base", config)
    logger.success(f"Saved {game_graph=} successfully.")

    start = time.perf_counter()
    solution = place_traps_and_fakes_greedy(game_graph, config)
    end = time.perf_counter()
    logger.success(f"Decoy placement completed.")
    logger.info(f"Time required to place decoys: {end - start} seconds.")

    # Extract solution graph. Save it, log it.
    sol_graph = solution.solution()
    sol_graph.save(os.path.join(directory, f'{exp_name}_solution.ggraph'))
    write_dot_file(sol_graph, "sol_graph", config)

    if config['graph']['save_png']:
        game_graph.to_png(os.path.join(directory, f'{exp_name}_base.png'),
                          nlabel=["state", "node_winner"],
                          elabel=["input", "edge_winner"])

    # Return variables of interest
    exec_time = end - start
    ram_used = None
    vod = sol_graph["vod"]
    return exec_time, ram_used, vod


def run_tom_and_jerry_experiment(experiment_cfg_dict: dict, game_cfg_path: str):
    # Create base graph
    tom_jerry_game = TomJerryGame(game_config=game_cfg_path)
    arbitrary_state = random.choice(tom_jerry_game.states())
    tom_jerry_game.initialize(arbitrary_state)
    game_graph = tom_jerry_game.graphify(pointed=False)

    # Logging and saving graph
    directory = experiment_cfg_dict['directory']
    exp_name = experiment_cfg_dict['name']
    if experiment_cfg_dict['graph']['save']:
        game_graph.save(os.path.join(directory, f'{exp_name}_base.ggraph'))
    if experiment_cfg_dict['graph']['save_png']:
        game_graph.to_png(os.path.join(directory, f'{exp_name}_base.png'), nlabel=["state"], elabel=["input"])
    logger.success(f"Saved base game graph {game_graph=} successfully.")

    # Create mapping from arena points to game states
    arena2state = dict()
    for node in game_graph.nodes():
        cell_tom, cell_jerry, states_door, player_turn = game_graph["state"][node]
        if cell_jerry not in arena2state:
            arena2state[cell_jerry] = []
        arena2state[cell_jerry].append(node)

    # Solve base game
    logger.info(f"Solving base game...")
    swin_game = solve_base_game(game_graph, tom_jerry_game, cfg_dict=experiment_cfg_dict)
    logger.info(f"Solved base game {swin_game=} successfully.")
    # Construct hypergame graph (Def. 6, in draft as of 4 Apr. 2023)
    hgame_graph = gen_hypergame(game_graph, swin_game)
    logger.info(f"Constructed hypergame graph {hgame_graph=} successfully.")
    path = os.path.join(experiment_cfg_dict['directory'], f"{experiment_cfg_dict['name']}_hgame.ggraph")
    hgame_graph.save(path)
    write_dot_file(hgame_graph, "hgame", experiment_cfg_dict)
    logger.info(f"Saved hypergame graph {hgame_graph=} successfully.")
    # Check for games where no traps can be placed
    assert len(set(hgame_graph.nodes()) - swin_game.get_final_states()) > 0, \
        f"Hypergame {hgame_graph} has no states where traps can be allocated"
    # Allocate decoys
    start = time.perf_counter()
    solution = place_decoys(hgame_graph, experiment_cfg_dict, arena2state)
    logger.success(f"Decoy placement completed.")
    end = time.perf_counter()
    logger.info(f"Time required to place decoys: {end - start} seconds.")

    decoys = solution.deception_dict["decoys"]
    print(f"{decoys=}")


def main():
    # Load configuration file
    config = cfg.process_cfg_file("configurations/config1.json")
    logger.success("Configuration loaded successfully.")

    # cProfile.run(run_experiment(config), "./out/50_3_hybrid_profile.txt")
    # prof = cProfile.Profile()
    # prof.runctx('run_experiment(config)', globals(), {'config': config})
    # prof.dump_stats('50_3_hybrid_profile.prof')
    exec_time, ram_used, vod = run_mixed_experiment(config)
    print(exec_time, ram_used, vod)
    # logger.success(f"Finished experiment config:{config['name']} with {exec_time=} sec, {ram_used=} bytes, and {vod=}.")

    # game_cfg_path = os.path.join("configurations", "game_2023_03_24_18_14.conf")
    # run_tom_and_jerry_experiment(experiment_cfg_dict=config, game_cfg_path=game_cfg_path)


if __name__ == '__main__':
    with logger.catch():
        main()
