import concurrent.futures
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

# Game Parameters
DIM = (5, 5)
GOAL_CELLS = [(0, 0), (0, 3), (3, 3)]
OBS_CELLS = [(0, 2), (0, 4), (1, 1), (4, 4)]
SENSOR_RNG = 1
P2_INIT = (0, 0)

# Define configuration
BASE_CONFIG = {
    "directory": "out",
    "filename": f"{pathlib.Path(__file__).name.split('.')[0]}",
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
        :param num_goals: int.
        """
        super(RndGridworld, self).__init__()
        self._dim = dim
        self._obs = obs if obs is not None else list()
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
        return [gw_util.GW_ACT_N, gw_util.GW_ACT_E, gw_util.GW_ACT_S, gw_util.GW_ACT_W,
                gw_util.GW_ACT_NE, gw_util.GW_ACT_SW]

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
            actions = [gw_util.GW_ACT_NE, gw_util.GW_ACT_SW]
            if (p2r, p2c) == (0, 0):
                actions.remove(gw_util.GW_ACT_SW)
            if (p2r, p2c) == (self._dim[0] - 1, self._dim[1] - 1):
                actions.remove(gw_util.GW_ACT_NE)

        # print(state, actions)
        return actions

    def delta(self, state, act):
        p1r, p1c, p2r, p2c, turn = state

        # Collision checking
        if (p1r, p1c) == (p2r, p2c):
            return state

        if turn == 1:
            p1r_prime, p1c_prime = gw_util.move((p1r, p1c), act)
            p1r_prime, p1c_prime = gw_util.bouncy_obstacle((p1r, p1c), [(p1r_prime, p1c_prime)], self._obs)[0]
            p1r_prime, p1c_prime = gw_util.bouncy_wall((p1r, p1c), [(p1r_prime, p1c_prime)], self._dim)[0]
            next_state = (p1r_prime, p1c_prime, p2r, p2c, 2)
        else:
            p2r_prime, p2c_prime = gw_util.move((p2r, p2c), act)
            p2r_prime, p2c_prime = gw_util.bouncy_obstacle((p2r, p2c), [(p2r_prime, p2c_prime)], self._obs)[0]
            p2r_prime, p2c_prime = gw_util.bouncy_wall((p2r, p2c), [(p2r_prime, p2c_prime)], self._dim)[0]
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
        return logic.ltl.ScLTL("F((g0 | g2) & F(g1))", atoms=self.atoms())

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
    game = RndGridworld(dim=DIM, goal_cells=GOAL_CELLS, sense_rng=SENSOR_RNG)

    # Iterate over initial states. Fix P2's state, P1's state variable. P1 plays first.
    p2r, p2c = P2_INIT
    for p1r, p1c in itertools.product(range(DIM[0]), range(DIM[1])):
        # Initialize the game
        game.initialize((p1r, p1c, p2r, p2c, 1))

        # Update the config
        config = BASE_CONFIG.copy()
        config["filename"] = f"{pathlib.Path(__file__).name.split('.')[0]}_{p1r}_{p1c}"
        dirpath = config["directory"] = os.path.join("out", f"{p1r}_{p1c}")
        if os.path.exists(dirpath) and os.path.isdir(dirpath):
            shutil.rmtree(dirpath)
        os.mkdir(dirpath)

        # Run the experiment
        exp.run_experiment(game, config=config)


if __name__ == "__main__":
    # main_single_inits()
    main_single_inits_multiprocessing()
