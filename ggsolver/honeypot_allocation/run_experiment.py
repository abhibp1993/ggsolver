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
                for t in range(1, 3)
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
        return state in [(r1, c2, r2, c2, t)
                         for r1, y1 in self.real_cheese
                         for r2 in range(self.rows)
                         for c2 in range(self.cols)
                         for t in range(1, 3)
                         if (r2, c2) not in self.obs
                         ]

    def equiv(self, cell):
        """ Returns states in which either P1 or P2 is at given cell """
        return {
                   cell + (r2, c2, t)
                   for r2 in range(self.rows)
                   for c2 in range(self.cols)
                   for t in range(1, 3)
                   if (r2, c2) not in self.obs
               } | {
                   (r1, c1) + cell + (t,)
                   for r1 in range(self.rows)
                   for c1 in range(self.cols)
                   for t in range(1, 3)
                   if (r1, c1) not in self.obs
               }


def gw1():
    obs = [(0, 2), (2, 2), (4, 2)]
    real_cheese = [(2, 4)]
    game = Gridworld(rows=7, cols=7, obs=obs, real_cheese=real_cheese)
    graph = game.graphify()

    # lines = []
    # for u, v, k in graph.edges():
    #     lines.append(f"{graph['state'][u]} -- {graph['input'][u, v, k]} --> {graph['state'][v]}")
    #
    # with open("out/trans.gm", "w") as file:
    #     for line in lines:
    #         file.write(line + "\n")

    return game, graph


if __name__ == '__main__':
    game, game_graph = gw1()

    fdir = "out/gw1_t2_f0"
    alloc = DecoyAllocator(game_graph, num_traps=2, num_fakes=0, debug=True, path=fdir)
    alloc.solve()
    alloc.save_pickle(fdir, filename="dswin_sol_graph")
    alloc.save_dot(fdir, filename="colored_graph")
