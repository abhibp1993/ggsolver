"""
Runs experiment and generates report based on cfg_dicturation files.
"""
import os.path
import cProfile
import networkx as nx
import matplotlib.pyplot as plt
import pygraphviz
from pygraphviz import graphviz

import ggsolver.decoy_alloc.process_config as cfg
import ggsolver.decoy_alloc.graph_generator as gen
import ggsolver.decoy_alloc.solvers as solvers
import ggsolver.decoy_alloc.models as models

import ggsolver.graph as ggraph
import ggsolver.models as gmodels
import ggsolver.dtptb as dtptb

import loguru

logger = loguru.logger
logger.remove()


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
    # svg_bytes = graph.pipe(format="svg")
    g = pygraphviz.AGraph(path)
    g.layout('dot')
    path = os.path.join(cfg_dict['directory'], f"{cfg_dict['name']}_{game_name}.svg")
    g.draw(path=path, format='svg')
    # with open(path, "wb") as file:
    #     file.write(svg_bytes)


def gen_game(cfg_dict: dict) -> dtptb.DTPTBGame:
    topology = cfg_dict['graph']['topology']
    if topology == "mesh":
        return gen.mesh(cfg_dict)
    elif topology == "ring":
        return gen.ring(cfg_dict)
    elif topology == "star":
        return gen.star(cfg_dict)
    elif topology == "tree":
        return gen.tree(cfg_dict)
    else:  # type == "hybrid":
        return gen.hybrid(cfg_dict)


def place_decoys(graph: ggraph.Graph, cfg_dict: dict) -> gmodels.Solver:
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
                                                num_decoys=num_traps,
                                                use_multiprocessing=cpu_count)

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
    print(f"Hidden edges {out_going_final_edges[0]}")

    hgame_graph = ggraph.SubGraph(game_graph, hidden_nodes=hidden_nodes, hidden_edges=hidden_edges)
    return hgame_graph


def run_experiment(config):
    # Generate base game
    # TODO. Make game, generate hypergame and then graphify.
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
    swin_game = dtptb.SWinReach(game_graph, player=2)
    swin_game.solve()
    path = os.path.join(config['directory'], f"{config['name']}_base.solution")
    swin_game.solution().save(path)

    logger.info(f"game.final:{[st for st in game.states() if game.final(st)]}.")
    logger.info(f"P1 SWin:{swin_game.winning_states(player=1)}, P2 SWin:{swin_game.winning_states(player=2)}")
    write_dot_file(swin_game.solution(), "base_game", config)
    logger.success(f"Solved {game_graph=} successfully.")

    # Construct hypergame graph (Def. 6, in draft as of 4 Apr. 2023)
    hgame_graph = gen_hypergame(game_graph, swin_game)
    path = os.path.join(config['directory'], f"{config['name']}_hgame.ggraph")
    hgame_graph.save(path)
    write_dot_file(hgame_graph, "hgame", config)
    logger.info(f"Constructed and saved {hgame_graph=} successfully.")

    # Allocate decoys
    solution = place_decoys(hgame_graph, config)
    logger.success(f"Decoy placement completed.")

    # Extract solution graph. Save it, log it.
    sol_graph = solution.solution()
    sol_graph.save(os.path.join(directory, f'{exp_name}_solution.ggraph'))
    write_dot_file(sol_graph, "sol_graph", config)

    if config['graph']['save_png']:
        game_graph.to_png(os.path.join(directory, f'{exp_name}_base.png'),
                          nlabel=["state", "node_winner"],
                          elabel=["input", "edge_winner"])

    # Generate reports and charts
    gen_reports(config)
    logger.warning("Report and charts is not yet implemented.")


def main():
    # Load configuration file
    config = cfg.process_cfg_file("configurations/config2.json")
    logger.success("Configuration loaded successfully.")

    # cProfile.run(run_experiment(config))
    run_experiment(config)


if __name__ == '__main__':
    with logger.catch():
        main()
