import logging
import os
import ggsolver.graph as ggraph
import ggsolver.dtptb.pgsolver as dtptb

logger = logging.getLogger(__name__)
CONFIG = {
    "directory": "out",
    "filename": "4by4_rng1_fixed",
}


def load_p1_solution():
    # Load game graph
    logger.info("Loading P1 game graph...")
    fpath = os.path.join(CONFIG["directory"], f"{CONFIG['filename']}_p1.ggraph")
    game_graph = ggraph.Graph().load(fpath)

    # Load solution
    logger.info("Loading P1 solution...")
    fpath = os.path.join(CONFIG["directory"], f"{CONFIG['filename']}_p1.dot")
    swin = dtptb.SWinReach(game_graph)
    swin.load_solution_from_dot(fpath)

    return swin


def load_p2_solution():
    # Load game graph
    logger.info("Loading P2 game graph...")
    fpath = os.path.join(CONFIG["directory"], f"{CONFIG['filename']}_p2.ggraph")
    game_graph = ggraph.Graph().load(fpath)

    # Load solution
    logger.info("Loading P2 solution...")
    fpath = os.path.join(CONFIG["directory"], f"{CONFIG['filename']}_p2.dot")
    swin = dtptb.SWinReach(game_graph)
    swin.load_solution_from_dot(fpath)

    return swin


if __name__ == '__main__':
    swin_p1 = load_p1_solution()
    swin_p2 = load_p2_solution()

    print(f"{len(swin_p1.winning_nodes(1))=}")
    print(f"{len(swin_p1.winning_nodes(2))=}")
    print(f"{len(swin_p2.winning_nodes(1))=}")
    print(f"{len(swin_p2.winning_nodes(2))=}")

    print(f"{swin_p1.winning_actions(swin_p1.node2state(0))=}")
    print(f"{swin_p2.winning_actions(swin_p2.node2state(0))=}")

