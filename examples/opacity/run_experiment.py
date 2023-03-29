import datetime
import time
import logging
from functools import partial
import ggsolver.dtptb.pgsolver as dtptb
import ggsolver.graph as graph
import models as opac_models
import os

logger = logging.getLogger(__name__)


DEFAULT_CONFIG = {
    "directory": "out",
    "filename": f"opacity_game_{datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}",
    "sensor_range": 1,
    "force_belief_graphify": False,
    "force_resolve": False,
}


def solve_p1game(game_graph: graph.Graph, path: str, filename: str, dot_file: str = None):
    # Define a reachability solver
    swin_reach_p1 = dtptb.SWinReach(game_graph, save_output=True, path=path, filename=filename + "_p1")
    logger.info("P1's SWinReach object created...")

    # Solve the reachability game
    if dot_file:
        logger.info(f"Loading solution of P1's game from {dot_file}.")
        swin_reach_p1.load_solution_from_dot(dot_file)
    else:
        swin_reach_p1.solve()

    return swin_reach_p1


def solve_p2game(game_graph: graph.Graph, p2final, path: str, filename: str, dot_file: str = None):
    # Generate final states
    # final = set(map(p2final, (game_graph["state"][uid] for uid in game_graph.nodes())))
    final = {game_graph["state"][uid] for uid in game_graph.nodes() if p2final(game_graph["state"][uid])}

    # When final is empty, there are no revealing winning states.
    if len(final) == 0:
        logger.info(f"There is no revealing winning states")

    # Create P2's solver
    swin_reach_p2 = dtptb.SWinReach(game_graph, final=final, save_output=True, path=path, filename=filename + "_p2")

    # Solve P2's game
    if dot_file:
        logger.info(f"Loading solution of P2's game from {dot_file}.")
        swin_reach_p2.load_solution_from_dot(dot_file)
    else:
        swin_reach_p2.solve()

    return swin_reach_p2


def run_experiment(game, config=None):
    if config is None:
        config = DEFAULT_CONFIG

    # Generate objective automaton
    aut = game.formula1().translate()
    aut_graph = aut.graphify()
    aut_graph.to_png(os.path.join(config["directory"], f"{config['filename']}_aut.png"),
                     nlabel=["state", "final"], elabel=["input"])

    # Generate and save the base game
    base_graph = game.graphify()
    base_graph.save(os.path.join(config["directory"], f"{config['filename']}_base.ggraph"), overwrite=True)

    # Define the belief game
    belief_game = opac_models.BeliefGame(game, aut)
    belief_game.initialize(belief_game.init_state())

    # Define P2's final state function
    p2final = partial(belief_game.final_p2)

    # If game is saved, load it. Else graphify it.
    fpath = os.path.join(config["directory"], f"{config['filename']}.ggraph")
    if os.path.exists(fpath) and not config["force_belief_graphify"]:
        game_graph = graph.Graph.load(fpath)
        logger.info(f"Loaded existing game graph from {fpath}...")

    else:
        # Graphify belief fame
        start = time.perf_counter()
        game_graph = belief_game.graphify(pointed=True)
        end = time.perf_counter()
        logger.info(f"Time for graphification: {end - start} seconds.")

        # Save the game.
        game_graph.save(fpath)
        logger.info(f"Saved the graphified belief game graph at {fpath}...")

    # Solve P1's game
    fpath = os.path.join(config["directory"], f"{config['filename']}_p1.dot")
    if os.path.exists(fpath) and not config["force_resolve"]:
        logger.info(f"Loading P1's game solution from {fpath}...")
        swin_reach_p1 = solve_p1game(game_graph, dot_file=fpath,
                                     path=config["directory"], filename=config["filename"])
        logger.info(f"Loaded P1's game solution from {fpath}.")
    else:
        logger.info(f"Solving P1 game from scratch...")
        start = time.perf_counter()
        swin_reach_p1 = solve_p1game(game_graph, path=config["directory"], filename=config["filename"])
        end = time.perf_counter()
        logger.info(f"Solution time for P1's game: {end - start} seconds.")

    fpath = os.path.join(config["directory"], f"{config['filename']}_p2.dot")
    if os.path.exists(fpath) and not config["force_resolve"]:
        logger.info(f"Loading P2's game solution from {fpath}...")
        swin_reach_p2 = solve_p2game(game_graph, p2final, dot_file=fpath,
                                     path=config["directory"], filename=config["filename"])
        logger.info(f"Loaded P2's game solution from {fpath}.")
    else:
        logger.info(f"Solving P2 game from scratch...")
        start = time.perf_counter()
        swin_reach_p2 = solve_p2game(game_graph, p2final,
                                     path=config["directory"], filename=config["filename"])
        end = time.perf_counter()
        logger.info(f"Solution time for P2's game: {end - start} seconds.")

    # Save the generated solutions
    fpath = os.path.join(config["directory"], f"{config['filename']}_p1.solution")
    swin_reach_p1.solution().save(fpath, overwrite=True)
    logger.info(f"Saved P1's game solution in '{fpath}'")

    fpath = os.path.join(config["directory"], f"{config['filename']}_p2.solution")
    swin_reach_p2.solution().save(fpath, overwrite=True)
    logger.info(f"Saved P2's game solution in '{fpath}'")