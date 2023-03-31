import datetime
import time
import logging
from functools import partial
import ggsolver.dtptb.pgsolver as dtptb
import ggsolver.graph as graph
import models as opac_models
import os

LOGGER = logging.getLogger(__name__)


DEFAULT_CONFIG = {
    "directory": "out",
    "filename": f"opacity_game_{datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}",
    "sensor_range": 1,
    "force_belief_graphify": False,
    "force_resolve": False,
}


def solve_p1game(game_graph: graph.Graph, path: str, filename: str, dot_file: str = None, logger=LOGGER):
    # Extract game init state. That's ID of process.
    init_state = game_graph["init_state"][0]

    # Define a reachability solver
    swin_reach_p1 = dtptb.SWinReach(game_graph, save_output=True, path=path, filename=filename + "_p1")
    logger.info(f"Game({init_state}):: P1's SWinReach object created...")

    # Solve the reachability game
    if dot_file:
        logger.info(f"Game({init_state}):: Loading solution of P1's game from {dot_file}.")
        swin_reach_p1.load_solution_from_dot(dot_file)
    else:
        swin_reach_p1.solve()

    return swin_reach_p1


def solve_p2game(game_graph: graph.Graph, p2final, path: str, filename: str, dot_file: str = None, logger=LOGGER):
    # Extract game init state. That's ID of process.
    init_state = game_graph["init_state"][0]

    # Generate final states
    # final = set(map(p2final, (game_graph["state"][uid] for uid in game_graph.nodes())))
    final = {game_graph["state"][uid] for uid in game_graph.nodes() if p2final(game_graph["state"][uid])}

    # When final is empty, there are no revealing winning states.
    if len(final) == 0:
        logger.info(f"Game({init_state}):: There is no revealing winning states")

    # Create P2's solver
    swin_reach_p2 = dtptb.SWinReach(game_graph, final=final, save_output=True, path=path, filename=filename + "_p2")

    # Solve P2's game
    if dot_file:
        logger.info(f"Game({init_state}):: Loading solution of P2's game from {dot_file}.")
        swin_reach_p2.load_solution_from_dot(dot_file)
    else:
        swin_reach_p2.solve()

    return swin_reach_p2


def run_experiment(game, game_init_set=None, config=None, logger=LOGGER):
    # Extract initial state of game. This defines the process.
    init_state = game.init_state()
    if config is None:
        config = DEFAULT_CONFIG
    logger.info(f"Config: {config}")

    # Generate objective automaton
    aut = game.formula1().translate()
    aut_graph = aut.graphify()
    aut_graph.to_png(os.path.join(config["directory"], f"{config['filename']}_aut.png"),
                     nlabel=["state", "final"], elabel=["input"])
    logger.info(f"Game({init_state}):: ScLTL({game.formula1()}) translated successfully...")

    # Generate and save the base game
    base_graph = game.graphify(pointed=True)
    base_graph.save(os.path.join(config["directory"], f"{config['filename']}_base.ggraph"), overwrite=True)
    logger.info(f"Game({game.init_state()}):: Base graph graphified successfully...")

    # Define the belief game
    belief_game = opac_models.BeliefGame(game, aut)
    belief_game_init_set = set()
    if game_init_set is None:
        s0 = belief_game.init_state()
        belief_game_init_set.add(s0)
    else:
        for st_ in game_init_set:
            game.initialize(st_)
            s0 = belief_game.init_state()
            belief_game_init_set.add(s0)
    logger.info(f"Game({game.init_state()}):: {belief_game_init_set=}")

    # Define P2's final state function
    p2final = partial(belief_game.final_p2)

    # If game is saved, load it. Else graphify it.
    fpath = os.path.join(config["directory"], f"{config['filename']}.ggraph")
    if os.path.exists(fpath) and not config["force_belief_graphify"]:
        game_graph = graph.Graph.load(fpath)
        logger.info(f"Game({init_state}):: Loaded existing game graph from {fpath}...")

    else:
        # Graphify belief fame
        start = time.perf_counter()
        print(f"belief_game.graphify(pointed=True, init_set={belief_game_init_set})")
        game_graph = belief_game.graphify(pointed=True, init_set=belief_game_init_set)
        end = time.perf_counter()
        logger.info(f"Game({init_state}):: Time for graphification: {end - start} seconds.")

        # Save the game.
        game_graph.save(fpath)
        logger.info(f"Game({init_state}):: Saved the graphified belief game graph at {fpath}...")

    # Solve P1's game
    fpath = os.path.join(config["directory"], f"{config['filename']}_p1.dot")
    if os.path.exists(fpath) and not config["force_resolve"]:
        logger.info(f"Game({init_state}):: Loading P1's game solution from {fpath}...")
        swin_reach_p1 = solve_p1game(game_graph, dot_file=fpath,
                                     path=config["directory"], filename=config["filename"], logger=logger)
        logger.info(f"Game({init_state}):: Loaded P1's game solution from {fpath}.")
    else:
        logger.info(f"Game({init_state}):: Solving P1 game from scratch...")
        start = time.perf_counter()
        swin_reach_p1 = solve_p1game(game_graph, path=config["directory"], filename=config["filename"], logger=logger)
        end = time.perf_counter()
        logger.info(f"Game({init_state}):: Solution time for P1's game: {end - start} seconds.")

    # Solve P2's game
    fpath = os.path.join(config["directory"], f"{config['filename']}_p2.dot")
    if os.path.exists(fpath) and not config["force_resolve"]:
        logger.info(f"Game({init_state}):: Loading P2's game solution from {fpath}...")
        swin_reach_p2 = solve_p2game(game_graph, p2final, dot_file=fpath,
                                     path=config["directory"], filename=config["filename"], logger=logger)
        logger.info(f"Game({init_state}):: Loaded P2's game solution from {fpath}.")
    else:
        logger.info(f"Game({init_state}):: Solving P2 game from scratch...")
        start = time.perf_counter()
        swin_reach_p2 = solve_p2game(game_graph, p2final,
                                     path=config["directory"], filename=config["filename"], logger=logger)
        end = time.perf_counter()
        logger.info(f"Game({init_state}):: Solution time for P2's game: {end - start} seconds.")

    # Save the generated solutions
    fpath = os.path.join(config["directory"], f"{config['filename']}_p1.solution")
    swin_reach_p1.solution().save(fpath, overwrite=True)
    logger.info(f"Game({init_state}):: Saved P1's game solution in '{fpath}'")

    fpath = os.path.join(config["directory"], f"{config['filename']}_p2.solution")
    swin_reach_p2.solution().save(fpath, overwrite=True)
    logger.info(f"Game({init_state}):: Saved P2's game solution in '{fpath}'")
