"""
Generate a random gridworld arena of size N_ROWS, N_COLS.
    - Actions: NESW
    - Obstacles: N_OBS (selected randomly)
    - Initial State: Choose randomly
    - Atoms: g1, g2
    - Labeling: Select goals randomly.
    - Formula: F(g1) & F(g2)
    - Sensor: If P1 within P2's sensor range, P1's location is known. Else, unknown.
    - Observation: Generated based on sensor.
"""
import itertools
import os
import random
import time

from scipy.spatial.distance import cityblock
from time import perf_counter

import ggsolver.gridworld.util as util
import ggsolver.dtptb.pgsolver as dtptb
import ggsolver.logic as logic
import ggsolver.graph as graph
import ggsolver.models as models

import models as mod_opacity
import logging


# Size of gridworld
DIM = (4, 4)
# Goal cells
GOAL_CELLS = [(0, 3), (3, 3)]
# Sensor range
SENSOR_RANGE = 1
# Output file name
DIRECTORY = "out"
FILENAME = "4by4_rng1_fixed"

logging.basicConfig(level=logging.DEBUG, filename=os.path.join(DIRECTORY, f"{FILENAME}.log"))


class RndGridworld(mod_opacity.Arena):
    GRAPH_PROPERTY = mod_opacity.Arena.GRAPH_PROPERTY.copy()

    def __init__(self, dim, goal_cells, obs=None, actions=None, init_state=None, sense_rng=2):
        """

        :param dim: (rows, col)
        :param obs: not processed yet!
        :param actions: see gridworld.utils package.
        :param init_state: obvious!
        :param sense_rng: int. Manhattan distance
        :param num_goals: int.
        """
        super(RndGridworld, self).__init__()
        self._dim = dim
        self._obs = obs
        self._actions = actions
        self._init_state = init_state
        self._sense_rng = sense_rng
        self._goal_cells = goal_cells
        self._num_goals = len(self._goal_cells)

    def states(self):
        """
        state: (p1r, p1c, p2r, p2c, turn)
        :return:
        """
        r_max, c_max = self._dim
        return itertools.product(range(r_max), range(c_max), range(r_max), range(c_max), [1, 2])

    def turn(self, state):
        return state[4]

    def actions(self):
        if self._actions is not None:
            return self._actions
        return [util.GW_ACT_N, util.GW_ACT_E, util.GW_ACT_S, util.GW_ACT_W]

    def delta(self, state, act):
        # TODO. No obstacle for now.
        p1r, p1c, p2r, p2c, turn = state
        next_state = state

        if turn == 1:
            p1r_prime, p1c_prime = util.move((p1r, p1c), act)
            if 0 <= p1r_prime < self._dim[0] and 0 <= p1c_prime < self._dim[1]:
                next_state = (p1r_prime, p1c_prime, p2r, p2c, 2)
        else:
            p2r_prime, p2c_prime = util.move((p2r, p2c), act)
            if 0 <= p2r_prime < self._dim[0] and 0 <= p2c_prime < self._dim[1]:
                next_state = (p1r, p1c, p2r_prime, p2c_prime, 1)

        return next_state

    def atoms(self):
        return [f"g{idx}" for idx in range(self._num_goals)]

    def label(self, state):
        if state[0:2] in self._goal_cells:
            idx = self._goal_cells.index(state[0:2])
            return [f"g{idx}"]
        return []

    def formula1(self):
        objective = [f"Fg{idx}" for idx in range(self._num_goals)]
        return logic.ltl.ScLTL(" & ".join(objective), atoms=self.atoms())

    def attacker_observation(self, state, act, next_state):
        p1r, p1c, p2r, p2c, turn = state
        p1r_prime, p1c_prime, p2r_prime, p2c_prime, turn_prime = next_state

        if turn == 1:
            if cityblock([p2r, p2c], [p1r_prime, p1c_prime]) <= self._sense_rng:
                return f"o1:{(p1r_prime, p1c_prime)}" + f"o2:{(p2r, p2c)}"
            else:
                return f"o1:{(p1r_prime,)}" + f"o2:{(p2r, p2c)}"

        else:
            if cityblock([p2r_prime, p2c_prime], [p1r, p1c]) <= self._sense_rng:
                return f"o1:{(p1r, p1c)}" + f"o2:{(p2r_prime, p2c_prime)}"
            else:
                return f"o1:{(p1r,)}" + f"o2:{(p2r_prime, p2c_prime)}"

    @models.register_property(GRAPH_PROPERTY)
    def goal_cells(self):
        return self._goal_cells


def solve(game: mod_opacity.BeliefGame, game_name: str):
    # Graphify the game
    if os.path.exists("out/" + game_name):
        game_graph = graph.Graph.load("out/" + game_name)
        print("Loaded existing game graph.")
    else:
        game_graph = game.graphify(pointed=True)
        game_graph.save("out/" + game_name)
        print("graphify done.")

    # Define a reachability solver (see dtptb.solvers.SWinReach)
    # swin_reach_p1 = dtptb.SWinReach(game_graph, save_output=True)
    # print("P1's SWinReach object created")

    # # Solve the safety game.
    # t1_start = perf_counter()
    # # swin_reach_p1.solve()
    # swin_reach_p1.process_pgsolver_dot(dot_file="out/pgzlk_2023_03_28_17_42_46.dot")
    # t1_stop = perf_counter()
    # print("Elapsed time during the calculation for P1 in s:",
    #       t1_stop - t1_start, 's')
    #
    # solution = swin_reach_p1._solution
    # solution.save("out/solution_1.solution")
    # print("P1's SWinReach object solved..")

    # print(f"{len(swin_reach_p1.winning_states(1))=}")

    # Define reachability solver for P2 (see Thm. 2)
    # final = {game_graph["state"] for uid in game_graph.nodes() if game.final_p2(game_graph["state"][uid])}
    final = set()
    for uid in game_graph.nodes():
        if game.final_p2(game_graph["state"][uid]):
            final.add(game_graph["state"][uid])

    if len(final) == 0:
        print("There is no revealing winning states")

    else:
        swin_reach_p2 = dtptb.SWinReach(game_graph, final=final, save_output=True)
        print("P2's SWinReach object created")

        # Solve the safety game.
        t2_start = perf_counter()
        # swin_reach_p2.solve()
        swin_reach_p2.process_pgsolver_dot(dot_file="out/pgzlk_2023_03_28_17_57_28.dot")
        t2_stop = perf_counter()
        print("Elapsed time during the calculation for P2 in s:",
              t2_start - t2_stop, 's')

        # solution = swin_reach_p2.solution()
        # solution.save("out/solution_2.solution")

        solution = swin_reach_p2._solution
        solution.save("out/solution_2.solution")
        print("P2's SWinReach object solved..")

        print(f"{len(swin_reach_p2.winning_states(1))=}")

    # Return solution to reachability game.
    return swin_reach_p2.winning_actions()


def main():
    # Instantiate random game here
    game = RndGridworld(dim=(4, 4), goal_cells=[(0, 3), (3, 3)], sense_rng=1)
    game.initialize((0, 0, 3, 0, 1))

    # Generate objective automaton
    aut = game.formula1().translate()
    aut_graph = aut.graphify()
    aut_graph.to_png("out/aut_graph.png", nlabel=["state", "final"], elabel=["input"])

    # Generate and save the base game
    base_graph = game.graphify()
    base_graph.save("out/base_game.gm", overwrite=True)

    # Define the belief game
    belief_game = mod_opacity.BeliefGame(game, aut)
    belief_game.initialize(belief_game.init_state())

    # Solve the game
    start = time.perf_counter()
    solve(belief_game, game_name=FILENAME))
    end = time.perf_counter()
    print(f"Total solution time: {end - start}")


if __name__ == "__main__":
    main()
