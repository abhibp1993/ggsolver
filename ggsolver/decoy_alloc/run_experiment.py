"""
Runs experiment and generates report based on cfg_dicturation files.
"""
import os.path

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
                                                     fname=cfg_dict['name']
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


def main():
    # Load configuration file
    config = cfg.process_cfg_file("configurations/config1.json")
    logger.success("Configuration loaded successfully.")

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
    swin_game = dtptb.SWinReach(game_graph, p=2)
    swin_game.solve()
    path = os.path.join(config['directory'], f"{config['name']}_base.solution")
    swin_game.solution().save(path)
    logger.success(f"Solved {game_graph=} successfully.")

    # Construct hypergame graph (Def. 6, in draft as of 4 Apr. 2023)
    hgame_graph = gen_hypergame(game_graph, swin_game, config)
    path = os.path.join(config['directory'], f"{config['name']}_hgame.ggraph")
    hgame_graph.save(path)
    logger.info(f"Constructed and saved {hgame_graph=} successfully.")

    # Allocate decoys
    solution = place_decoys(hgame_graph, config)
    logger.success(f"Decoy placement completed.")

    # Extract solution graph. Save it, log it.
    sol_graph = solution.solution()
    sol_graph.save(os.path.join(directory, f'{exp_name}_solution.ggraph'))

    if config['graph']['save_png']:
        game_graph.to_png(os.path.join(directory, f'{exp_name}_base.png'),
                          nlabel=["state", "node_winner"],
                          elabel=["input", "edge_winner"])

    # Generate reports and charts
    gen_reports(config)
    logger.warning("Report and charts is not yet implemented.")


if __name__ == '__main__':
    with logger.catch():
        main()
