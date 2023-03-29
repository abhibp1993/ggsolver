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

from functools import partial

from scipy.spatial.distance import cityblock
import ggsolver.gridworld.util as util
import ggsolver.dtptb.pgsolver as dtptb
import ggsolver.logic as logic
import ggsolver.graph as graph
import ggsolver.models as models
import ggsolver.util

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
# Force regraphification of belief game?
FORCE_REGEN = False
# Force games solutions to be recomputed, in case they exist.
FORCE_RESOLVE = False


# Set up logging
logging.basicConfig(level=logging.INFO)


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


def solve_p1game(game_graph: graph.Graph, dot_file: str = None):
    # Define a reachability solver
    swin_reach_p1 = dtptb.SWinReach(game_graph, save_output=True, path=DIRECTORY, filename=f"{FILENAME}_p1")
    logging.info("P1's SWinReach object created...")

    # Solve the reachability game
    if dot_file:
        logging.info(f"Loading solution of P1's game from {dot_file}.")
        swin_reach_p1.load_solution_from_dot(dot_file)
    else:
        t1_start = time.perf_counter()
        swin_reach_p1.solve()
        t1_stop = time.perf_counter()
        logging.info(f"Time for solving P1's game: {t1_stop - t1_start} seconds")

    return swin_reach_p1


def solve_p2game(game_graph: graph.Graph, p2final, dot_file: str = None):
    # Generate final states
    # final = set(map(p2final, (game_graph["state"][uid] for uid in game_graph.nodes())))
    final = {game_graph["state"][uid] for uid in game_graph.nodes() if p2final(game_graph["state"][uid])}

    # When final is empty, there are no revealing winning states.
    if len(final) == 0:
        logging.info(f"There is no revealing winning states")

    # Create P2's solver
    swin_reach_p2 = dtptb.SWinReach(game_graph, final=final, save_output=True, path=DIRECTORY, filename=f"{FILENAME}_p2")

    # Solve P2's game
    if dot_file:
        logging.info(f"Loading solution of P2's game from {dot_file}.")
        swin_reach_p2.load_solution_from_dot(dot_file)
    else:
        t2_start = time.perf_counter()
        swin_reach_p2.solve()
        t2_stop = time.perf_counter()
        logging.info(f"Time for solving P2's game: {t2_stop - t2_start} seconds")

    return swin_reach_p2


def main():
    # Instantiate random game here
    game = RndGridworld(dim=(4, 4), goal_cells=[(0, 3), (3, 3)], sense_rng=1)
    game.initialize((0, 0, 3, 0, 1))

    # Generate objective automaton
    aut = game.formula1().translate()
    aut_graph = aut.graphify()
    aut_graph.to_png(os.path.join(DIRECTORY, f"{FILENAME}_aut.png"), nlabel=["state", "final"], elabel=["input"])

    # Generate and save the base game
    base_graph = game.graphify()
    base_graph.save(os.path.join(DIRECTORY, f"{FILENAME}_base.ggraph"), overwrite=True)

    # Define the belief game
    belief_game = mod_opacity.BeliefGame(game, aut)
    belief_game.initialize(belief_game.init_state())

    # Define P2's final state function
    p2final = partial(belief_game.final_p2)

    # If game is saved, load it. Else graphify it.
    fpath = os.path.join(DIRECTORY, f"{FILENAME}.ggraph")
    if os.path.exists(fpath) and not FORCE_REGEN:
        game_graph = graph.Graph.load(fpath)
        logging.info(f"Loaded existing game graph from {fpath}...")

    else:
        # Graphify belief fame
        start = time.perf_counter()
        game_graph = belief_game.graphify(pointed=True)
        end = time.perf_counter()
        logging.info(f"Time for graphification: {end - start} seconds.")

        # Save the game.
        game_graph.save(fpath)
        logging.info(f"Saved the graphified belief game graph at {fpath}...")

    # Solve P1's game
    fpath = os.path.join(DIRECTORY, f"{FILENAME}_p1.dot")
    if os.path.exists(fpath) and not FORCE_RESOLVE:
        logging.info(f"Loading P1's game solution from {fpath}...")
        swin_reach_p1 = solve_p1game(game_graph, dot_file=fpath)
        logging.info(f"Loaded P1's game solution from {fpath}.")
    else:
        logging.info(f"Solving P1 game from scratch...")
        start = time.perf_counter()
        swin_reach_p1 = solve_p1game(game_graph)
        end = time.perf_counter()
        logging.info(f"Solution time for P1's game: {end - start} seconds.")

    fpath = os.path.join(DIRECTORY, f"{FILENAME}_p2.dot")
    if os.path.exists(fpath) and not FORCE_RESOLVE:
        logging.info(f"Loading P2's game solution from {fpath}...")
        swin_reach_p2 = solve_p2game(game_graph, p2final, fpath)
        logging.info(f"Loaded P2's game solution from {fpath}.")
    else:
        logging.info(f"Solving P2 game from scratch...")
        start = time.perf_counter()
        swin_reach_p2 = solve_p2game(game_graph, p2final)
        end = time.perf_counter()
        logging.info(f"Solution time for P2's game: {end - start} seconds.")

    # Save the generated solutions
    fpath = os.path.join(DIRECTORY, f"{FILENAME}_p1.solution")
    swin_reach_p1.solution().save(fpath, overwrite=True)
    logging.info(f"Saved P1's game solution in '{fpath}'")

    fpath = os.path.join(DIRECTORY, f"{FILENAME}_p2.solution")
    swin_reach_p2.solution().save(fpath, overwrite=True)
    logging.info(f"Saved P2's game solution in '{fpath}'")


if __name__ == "__main__":
    main()
