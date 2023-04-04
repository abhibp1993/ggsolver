"""
Runs experiment and generates report based on cfg_dicturation files.
"""
import os.path

import ggsolver.decoy_alloc.process_config as cfg
import ggsolver.decoy_alloc.graph_generator as gen
import ggsolver.decoy_alloc.solvers as solvers

import ggsolver.graph as ggraph
import ggsolver.models as models

import loguru

logger = loguru.logger
logger.remove()


def gen_graph(cfg_dict: dict) -> ggraph.Graph:
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


def place_decoys(graph: ggraph.Graph, cfg_dict: dict) -> models.Solver:
    type_ = cfg_dict['type']
    num_traps = cfg_dict['max_traps']
    num_fakes = cfg_dict['max_fakes']
    use_multiprocessing = cfg_dict['use_multiprocessing']

    if type_ == "enumerative" and num_traps > 0 and num_fakes == 0:
        solution = solvers.EnumerativeTrapsAllocator(graph=graph,
                                                     num_decoys=num_traps,
                                                     use_multiprocessing=use_multiprocessing)

    elif type_ == "greedy" and num_traps > 0 and num_fakes == 0:
        solution = solvers.GreedyTrapsAllocator(graph=graph,
                                                num_decoys=num_traps,
                                                use_multiprocessing=use_multiprocessing)

    elif type_ == "enumerative" and num_traps == 0 and num_fakes > 0:
        solution = solvers.EnumerativeFakesAllocator(graph=graph,
                                                     num_decoys=num_fakes,
                                                     use_multiprocessing=use_multiprocessing)

    elif type_ == "greedy" and num_traps == 0 and num_fakes > 0:
        solution = solvers.GreedyFakesAllocator(graph=graph,
                                                num_decoys=num_fakes,
                                                use_multiprocessing=use_multiprocessing)

    elif type_ == "enumerative" and num_traps > 0 and num_fakes > 0:
        solution = solvers.EnumerativeMixedAllocator(graph=graph,
                                                     num_decoys=(num_traps, num_fakes),
                                                     use_multiprocessing=use_multiprocessing
                                                     )

    else:  # type_ == "greedy" and num_traps == 0 and num_fakes > 0:
        solution = solvers.GreedyMixedAllocator(graph=graph,
                                                num_decoys=(num_traps, num_fakes),
                                                use_multiprocessing=use_multiprocessing)

    # Solve the game
    solution.solve()
    return solution


if __name__ == '__main__':
    # Load configuration file
    config = cfg.process_cfg_file("configurations/config1.json")
    logger.success("Configuration loaded successfully.")

    # Generate graph
    game_graph = gen_graph(config)
    logger.success(f"Generated {game_graph=} successfully.")

    # Logging and saving graph
    directory = config['directory']
    exp_name = config['name']
    if config['graph']['save']:
        game_graph.save(os.path.join(directory, f'{exp_name}_base.ggraph'))

    if config['graph']['save_png']:
        game_graph.to_png(os.path.join(directory, f'{exp_name}_base.png'), nlabel=["state"], elabel=["input"])

    # Allocate decoys
    solution = place_decoys(game_graph, config)
    logger.success(f"Decoy placement completed.")

    # Extract solution graph. Save it, log it.
    sol_graph = solution.solution()
    sol_graph.save(os.path.join(directory, f'{exp_name}_solution.ggraph'))

    if config['graph']['save_png']:
        game_graph.to_png(os.path.join(directory, f'{exp_name}_base.png'),
                          nlabel=["state", "node_winner"],
                          elabel=["input", "edge_winner"])

    # Generate reports and charts
    logger.warning("Report and charts is not yet implemented.")
