"""
ACTUAL EXPERIMENT:
-----------------------
Arena is 5x5 grid with 4 obstacles.
P1 moves freely.
P2 is restricted in L-shaped region.
Goals placed in such a way that P1 wins if it starts within 2-steps from G0.
"""

import concurrent.futures
import logging
import os.path
import pathlib
import shutil

from scipy.spatial.distance import cityblock
import run_experiment as exp
import models as opac_models
import itertools
import ggsolver.gridworld.util as gw_util
import ggsolver.logic as logic
import ggsolver.models as gg_models

# from loguru import logger
import sys

logger = logging.getLogger(__name__)

sys.path.append('/home/ggsolver')
FILENAME = pathlib.Path(__file__).name.split('.')[0]

# logger.remove()
# logger.add(sys.stderr, format="[{level}]:: {message}", level="ERROR")
# logger.add(f"out/{FILENAME}.log", format="[{level}]:: {message}", level="DEBUG")

# Game Parameters
DIM = (5, 5)
# GOAL_CELLS = [(0, 4), (3, 1), (1, 3)]
GOAL_CELLS = [(0, 4), (1, 1), (3, 3)]
# GOAL_CELLS = [(4, 0), (1, 1), (3, 3)]
OBS_CELLS = []
SENSOR_RNG = 1
P2_INIT = (0, 0)
P2_WALKABLE = [
    (4, 0), (4, 1), (4, 2), (4, 3), (4, 4),
    (3, 0), (3, 4),
    (2, 0), (2, 4),
    (1, 0), (1, 4),
    (0, 0), (0, 1), (0, 2), (0, 3), (0, 4)
    ]
FORMULA = "F((g0) & F(g1 | g2))"


# Define configuration
BASE_CONFIG = {
    "directory": f"out/{FILENAME}",
    "filename": f"{FILENAME}",
    "force_belief_graphify": False,
    "force_resolve": False,
}


class RndGridworld(opac_models.Arena):
    GRAPH_PROPERTY = opac_models.Arena.GRAPH_PROPERTY.copy()

    def __init__(self, dim, goal_cells, obs=None, actions=None, init_state=None, sense_rng=2):
        """

        :param dim: (rows, col)
        :param obs: not processed yet!
        :param actions: see gridworld.utils package.
        :param init_state: obvious!
        :param sense_rng: int. Manhattan distance
        """
        super(RndGridworld, self).__init__()
        self._dim = dim
        self._p2_walkable = P2_WALKABLE
        self._obs = obs if obs is not None else list()
        self._actions = actions
        self._init_state = init_state
        self._sense_rng = sense_rng
        self._goal_cells = goal_cells
        self._num_goals = len(self._goal_cells)
        self._p2_acts = {
            (0, 0): [gw_util.GW_ACT_N, gw_util.GW_ACT_E],
            (0, 1): [gw_util.GW_ACT_W, gw_util.GW_ACT_E],
            (0, 2): [gw_util.GW_ACT_W, gw_util.GW_ACT_E],
            (0, 3): [gw_util.GW_ACT_W, gw_util.GW_ACT_E],
            (0, 4): [gw_util.GW_ACT_W, gw_util.GW_ACT_N],
            (1, 0): [gw_util.GW_ACT_N, gw_util.GW_ACT_S],
            (1, 4): [gw_util.GW_ACT_N, gw_util.GW_ACT_S],
            (2, 0): [gw_util.GW_ACT_N, gw_util.GW_ACT_S],
            (2, 4): [gw_util.GW_ACT_N, gw_util.GW_ACT_S],
            (3, 0): [gw_util.GW_ACT_N, gw_util.GW_ACT_S],
            (3, 4): [gw_util.GW_ACT_N, gw_util.GW_ACT_S],
            (4, 0): [gw_util.GW_ACT_S, gw_util.GW_ACT_E],
            (4, 1): [gw_util.GW_ACT_W, gw_util.GW_ACT_E],
            (4, 2): [gw_util.GW_ACT_W, gw_util.GW_ACT_E],
            (4, 3): [gw_util.GW_ACT_W, gw_util.GW_ACT_E],
            (4, 4): [gw_util.GW_ACT_W, gw_util.GW_ACT_S],

        }

    def states(self):
        """
        state: (p1r, p1c, p2r, p2c, turn)
        :return:
        """
        r_max, c_max = self._dim
        states = itertools.product(range(r_max), range(c_max), self._p2_walkable, [1, 2])
        return ((p1r, p1c, p2r, p2c, turn) for p1r, p1c, (p2r, p2c), turn in states)

    def turn(self, state):
        return state[4]

    def actions(self):
        if self._actions is not None:
            return self._actions
        return [gw_util.GW_ACT_N, gw_util.GW_ACT_E, gw_util.GW_ACT_S, gw_util.GW_ACT_W]

    def enabled_acts(self, state):
        p1r, p1c, p2r, p2c, turn = state

        if self.turn(state) == 1:
            actions = [gw_util.GW_ACT_N, gw_util.GW_ACT_E, gw_util.GW_ACT_S, gw_util.GW_ACT_W]
            if p1r == 0:
                actions.remove(gw_util.GW_ACT_S)
            if p1c == 0:
                actions.remove(gw_util.GW_ACT_W)
            if p1r == self._dim[0] - 1:
                actions.remove(gw_util.GW_ACT_N)
            if p1c == self._dim[1] - 1:
                actions.remove(gw_util.GW_ACT_E)

        else:  # self.turn(state) == 2:
            return self._p2_acts[p2r, p2c]

        # logger.debug(f"En({state}): {actions}")
        return actions

    def delta(self, state, act):
        p1r, p1c, p2r, p2c, turn = state

        # Collision checking
        if (p1r, p1c) == (p2r, p2c):
            logger.debug(f"{state} -- {act} -> {state}")
            return state

        if turn == 1:
            p1r_prime, p1c_prime = gw_util.move((p1r, p1c), act)
            p1r_prime, p1c_prime = gw_util.bouncy_obstacle((p1r, p1c), [(p1r_prime, p1c_prime)], self._obs)[0]
            p1r_prime, p1c_prime = gw_util.bouncy_wall((p1r, p1c), [(p1r_prime, p1c_prime)], self._dim)[0]
            next_state = (p1r_prime, p1c_prime, p2r, p2c, 2)
        else:
            p2r_prime, p2c_prime = gw_util.move((p2r, p2c), act)
            p2r_prime, p2c_prime = gw_util.bouncy_obstacle((p2r, p2c), [(p2r_prime, p2c_prime)], self._obs)[0]
            if (p2r_prime, p2c_prime) not in self._p2_walkable:
                p2r_prime, p2c_prime = p2r, p2c
            next_state = (p1r, p1c, p2r_prime, p2c_prime, 1)

        logger.debug(f"{state} -- {act} -> {next_state}")
        return next_state

    def atoms(self):
        return [f"g{idx}" for idx in range(self._num_goals)]

    def label(self, state):
        if state[0:2] in self._goal_cells:
            idx = self._goal_cells.index(state[0:2])
            return [f"g{idx}"]
        return []

    def formula1(self):
        return logic.ltl.ScLTL(FORMULA, atoms=self.atoms())

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

    @gg_models.register_property(GRAPH_PROPERTY)
    def goal_cells(self):
        return self._goal_cells


def main_single_inits_multiprocessing():
    # Instantiate random game here
    game = RndGridworld(dim=DIM, goal_cells=GOAL_CELLS, sense_rng=SENSOR_RNG, obs=OBS_CELLS)

    # Iterate over initial states. Fix P2's state, P1's state variable. P1 plays first.
    p2r, p2c = P2_INIT
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = list()
        for p1r, p1c in itertools.product(range(DIM[0]), range(DIM[1])):
            if (p1r, p1c) in OBS_CELLS:
                continue

            # Initialize the game
            s0 = (p1r, p1c, p2r, p2c, 1)
            game.initialize(s0)

            # Update the config
            config = BASE_CONFIG.copy()
            config["filename"] = f"{config['filename']}_{p1r}_{p1c}"
            dirpath = config["directory"] = os.path.join(config["directory"], f"{p1r}_{p1c}")
            if os.path.exists(dirpath) and os.path.isdir(dirpath):
                shutil.rmtree(dirpath)
            path_ = pathlib.Path(dirpath)
            path_.mkdir(parents=True)

            # Add argument
            futures.append(executor.submit(exp.run_experiment, game, {s0}, config.copy()))

        for future in concurrent.futures.as_completed(futures):
            print(f"{future}: {future.result()=}")


def main_single_inits():
    # Instantiate random game here
    game = RndGridworld(dim=DIM, goal_cells=GOAL_CELLS, sense_rng=SENSOR_RNG, obs=OBS_CELLS)

    # Iterate over initial states. Fix P2's state, P1's state variable. P1 plays first.
    p2r, p2c = P2_INIT
    for p1r, p1c in itertools.product(range(DIM[0]), range(DIM[1])):
        if (p1r, p1c) in OBS_CELLS:
            continue

        # Initialize the game
        game.initialize((p1r, p1c, p2r, p2c, 1))

        # Update the config
        config = BASE_CONFIG.copy()
        config["filename"] = f"{FILENAME}_{p1r}_{p1c}"
        dirpath = config["directory"] = os.path.join(config["directory"], f"{p1r}_{p1c}")
        if os.path.exists(dirpath) and os.path.isdir(dirpath):
            shutil.rmtree(dirpath)
        path_ = pathlib.Path(dirpath)
        path_.mkdir(parents=True)

        # Run the experiment
        exp.run_experiment(game, config=config)


if __name__ == "__main__":
    logger.info("loguru says hi!")
    # main_single_inits()
    main_single_inits_multiprocessing()
