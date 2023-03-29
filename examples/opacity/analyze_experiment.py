import itertools
import logging
import os
import random

import ggsolver.graph as ggraph
import ggsolver.dtptb.pgsolver as dtptb

logger = logging.getLogger(__name__)
CONFIG = {
    "directory": "out/multi_init_setup",
    "filename": "ex6_4x4grid_allinit_multi_init",
}

# Game Parameters
DIM = (4, 4)
GOAL_CELLS = [(0, 3), (3, 3)]
SENSOR_RNG = 1
P2_INIT = (3, 0)


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


def print_init_winners(win1):

    # Iterate over initial states. Fix P2's state, P1's state variable. P1 plays first.
    p2r, p2c = P2_INIT
    for p1r, p1c in itertools.product(range(DIM[0]), range(DIM[1])):
        s0 = (p1r, p1c, p2r, p2c, 1)
        q0 = 2
        v0 = (s0, q0, ((s0, q0), ))

        print(f"{v0}: {win1.state_winner(v0)=}")


def generate_play(win, v0):
    finals = []     # win.graph()["final"]
    curr_node = win.state2node(v0)
    path = []
    while True:
        win_edges = win.winning_edges(curr_node)
        edge = random.choice(win_edges)
        path.append(edge)
        dst = win.graph()["state"][edge[1]]
        curr_node = dst
        if dst in finals:
            break



if __name__ == '__main__':
    swin_p1 = load_p1_solution()
    swin_p2 = load_p2_solution()

    init_state = swin_p1.graph()["init_state"]

    print(f"{len(swin_p1.winning_nodes(1))=}")
    print(f"{len(swin_p1.winning_nodes(2))=}")
    print(f"{len(swin_p2.winning_nodes(1))=}")
    print(f"{len(swin_p2.winning_nodes(2))=}")

    print(f"{swin_p1.winning_actions(swin_p1.node2state(0))=}")
    print(f"{swin_p2.winning_actions(swin_p2.node2state(0))=}")

    print_init_winners(swin_p1)
    print_init_winners(swin_p2)

