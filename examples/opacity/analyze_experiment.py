import itertools
import logging
import os
import random

import ggsolver.graph as ggraph
import ggsolver.dtptb.pgsolver as dtptb

logger = logging.getLogger(__name__)
# CONFIG = {
#     "directory": "out/multiprocessing/0_0",
#     "filename": "ex6_4x4grid_allinit_0_0",
# }

# Game Parameters
DIM = (5, 5)
GOAL_CELLS = [(0, 0), (0, 4), (4, 4)]
# OBS_CELLS = [(0, 1), (0, 3), (1, 4), (1, 0), (2, 3), (2, 4), (3, 4), (3, 0), (4, 3), (4, 4)]
SENSOR_RNG = 1
P2_INIT = (0, 0)


# DIM = (5, 4)
# GOAL_CELLS = [(1, 3), (0, 2)]
# SENSOR_RNG = 1
# P2_INIT = (4, 0)


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


def print_init_winners(win1, winner):
    # Iterate over initial states. Fix P2's state, P1's state variable. P1 plays first.
    p2r, p2c = P2_INIT
    p1r = int(CONFIG["filename"][-3])
    p1c = int(CONFIG["filename"][-1])
    s0 = (p1r, p1c, p2r, p2c, 1)
    q0 = 1
    v0 = (s0, q0, ((s0, q0),))
    winner[v0] = win1.state_winner(v0)
    print(f"{v0}: {win1.state_winner(v0)=}")


def generate_play(win, v0, max_iter=float("inf")):
    finals = [uid for uid in win.graph().nodes() if win.graph()["final"][uid]]
    curr_node = win.state2node(v0)
    path = [v0]
    count = 0
    print(v0)
    while count < max_iter:
        win_edges = win.winning_edges(curr_node)
        win_actions = [win.graph()["input"][edge] for edge in win_edges]
        edge = random.choice(win_edges)
        edge = eval(input(f"Choose next: {list(zip(win_edges, win_actions))}"))
        act = win.graph()["input"][edge]
        path.append(act)
        dst = win.graph()["state"][edge[1]]
        curr_node = dst
        path.append(curr_node)
        curr_node = win.state2node(dst)
        if curr_node in finals:
            break
        print(act, win_actions)
        print(dst)
        count += 1
    return path


if __name__ == '__main__':
    # position_winners = dict()
    # for i, j in itertools.product(range(DIM[0]), range(DIM[1])):
    #     CONFIG = {
    #         "directory": f"out/ex14_5x5wumpus/{i}_{j}",
    #         "filename": f"ex14_5x5wumpus_{i}_{j}",
    #     }
    #
    #     try:
    #         # if i != 0 or j != 3:
    #         swin_p1 = load_p1_solution()
    #         # swin_p2 = load_p2_solution()
    #
    #         init_state = swin_p1.graph()["init_state"]
    #
    #         print(f"{len(swin_p1.winning_nodes(1))=}")
    #         print(f"{len(swin_p1.winning_nodes(2))=}")
    #         # print(f"{len(swin_p2.winning_nodes(1))=}")
    #         # print(f"{len(swin_p2.winning_nodes(2))=}")
    #
    #         print(f"{swin_p1.winning_actions(swin_p1.node2state(0))=}")
    #         # print(f"{swin_p2.winning_actions(swin_p2.node2state(0))=}")
    #
    #         print_init_winners(swin_p1, position_winners)
    #         # print_init_winners(swin_p2)
    #     except Exception as err:
    #         print(f"++++++++++++++++++ NO RESULT FOR {i, j} ++++++++++++++++++")
    #         print(err)
    #
    # from pprint import pprint
    # pprint(position_winners)

    CONFIG = {
        "directory": "out/ex14_5x5wumpus/0_1",
        "filename": "ex14_5x5wumpus_0_1",
    }

    # if i != 0 or j != 3:
    swin_p1 = load_p1_solution()
    init_state = swin_p1.graph()["init_state"]
    # swin_p2 = load_p2_solution()
    # init_state = swin_p2.graph()["init_state"]
    print("Start generating plays")

    play = generate_play(swin_p1, init_state)
    # play = generate_play(swin_p2, init_state, max_iter=50)
    for state in play:
        print(state)

    print(play)
