import ggsolver.graph as ggraph
import ggsolver.dtptb as dtptb
from ggsolver.gridworld.util import *
from ggsolver.honeypot_allocation.solvers import DecoyAllocator
from loguru import logger


class Gridworld(dtptb.DTPTBGame):
    def __init__(self, rows, cols, obs, real_cheese):
        super(Gridworld, self).__init__()
        self.rows = rows
        self.cols = cols
        self.obs = obs
        self.real_cheese = real_cheese

    def __str__(self):
        delta = {(st, a): self.delta(st, a) for st in self.states() for a in self.enabled_acts(st)}
        final = {st for st in self.states() if self.final(st)}
        return f"{self.states()=}\n" \
               f"{delta=}\n" \
               f"{final=}\n"

    def states(self):
        """ (r1, c1): tom (defender), (r2, c2): jerry (attacker) """
        return [(r1, c1, r2, c2, t)
                for r1 in range(self.rows)
                for c1 in range(self.cols)
                for r2 in range(self.rows)
                for c2 in range(self.cols)
                for t in [1, 2]
                if (r1, c1) not in self.obs and (r2, c2) not in self.obs
                ]

    def turn(self, state):
        return state[-1]

    def actions(self):
        return [GW_ACT_N, GW_ACT_E, GW_ACT_S, GW_ACT_W]

    def delta(self, state, act):
        turn = self.turn(state)

        if state[0:2] == state[2:4]:
            return state[0:4] + ((1,) if turn == 2 else (2,))

        if turn == 1:
            r, c = state[0:2]
        else:
            r, c = state[2:4]

        next_r, next_c = move((r, c), act)
        next_r, next_c = bouncy_wall((r, c), [(next_r, next_c)], (self.rows, self.cols))[0]
        next_r, next_c = bouncy_obstacle((r, c), [(next_r, next_c)], self.obs)[0]

        if turn == 1:
            return (next_r, next_c) + state[2:4] + (2,)
        else:
            return state[0:2] + (next_r, next_c) + (1,)

    def final(self, state):
        return state in [(r1, c1, r2, c2, t)
                         for r2, c2 in self.real_cheese
                         for r1 in range(self.rows)
                         for c1 in range(self.cols)
                         for t in range(1, 3)
                         if (r2, c2) not in self.obs and (r1, c1) not in self.obs
                         ]

    def jerry_equiv(self, cell):
        """ Returns states in which Jerry is at given cell """
        # return {
        #            cell + (r2, c2, t)
        #            for r2 in range(self.rows)
        #            for c2 in range(self.cols)
        #            for t in range(1, 3)
        #            if (r2, c2) not in self.obs
        #        } | \
        return {
            (r1, c1) + cell + (t,)
            for r1 in range(self.rows)
            for c1 in range(self.cols)
            for t in range(1, 3)
            if (r1, c1) not in self.obs
        }


def gw1():
    obs = [(0, 4), (2, 4), (4, 4), (6, 4)]
    real_cheese = [(2, 5)]
    game = Gridworld(rows=7, cols=7, obs=obs, real_cheese=real_cheese)
    graph = game.graphify()
    state2node = {graph['state'][node]: node for node in graph.nodes()}
    candidates = dict()
    for cell in {(r, c) for r in range(7) for c in range(7) if (r, c) not in set(obs) | set(real_cheese)}:
        candidates[cell] = {state2node[state] for state in game.jerry_equiv(cell)}

    return game, graph, candidates


def gw2():
    obs = [(0, 4), (2, 4), (4, 4), (6, 4)]
    real_cheese = [(1, 6), (4, 6)]
    game = Gridworld(rows=7, cols=7, obs=obs, real_cheese=real_cheese)
    graph = game.graphify()
    state2node = {graph['state'][node]: node for node in graph.nodes()}
    candidates = dict()
    for cell in {(r, c) for r in range(7) for c in range(7) if (r, c) not in set(obs) | set(real_cheese)}:
        candidates[cell] = {state2node[state] for state in game.jerry_equiv(cell)}

    return game, graph, candidates


def swin_run():
    obs = [(5, 5), (1, 0), (0, 1), (6, 5), (2, 3)]
    cheese = [(2, 1), (2, 0)]
    game = Gridworld(rows=7, cols=7, obs=obs, real_cheese=cheese)
    graph = game.graphify()
    state2node = {graph['state'][node]: node for node in graph.nodes()}
    candidates = dict()
    for cell in {(r, c) for r in range(7) for c in range(7) if (r, c) not in set(obs) | set(cheese)}:
        candidates[cell] = {state2node[state] for state in game.jerry_equiv(cell)}

    logger.info(f"{obs=}")
    logger.info(f"{cheese=}")
    logger.info(f"fakes={5}")
    logger.info(f"traps={0}")
    alloc = DecoyAllocator(graph, num_traps=0, num_fakes=5, candidates=candidates, debug=True, algo="greedy")
    alloc.solve()


if __name__ == '__main__':
    swin_run()
    # game, game_graph, candidates = gw2()
    # fdir = "out/gw2_t2_f0_enum"
    #
    # # # Greedy approach
    # # alloc = DecoyAllocator(game_graph, num_traps=0, num_fakes=2, candidates=candidates, debug=True, path=fdir)
    # # alloc.solve()
    # # alloc.save_pickle(fdir, filename="dswin_sol_graph")
    #
    # # Enumerative approach
    # alloc = DecoyAllocator(game_graph, num_traps=2, num_fakes=0, candidates=candidates, debug=True, path=fdir, algo="enumerative")
    # alloc.solve()
