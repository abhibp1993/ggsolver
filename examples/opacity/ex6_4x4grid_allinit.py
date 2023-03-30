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
DIM = (4, 4)
GOAL_CELLS = [(3, 0), (3, 3)]
OBS_CELLS = [(1, 0), (1, 1), (1, 2)]
SENSOR_RNG = 1
P2_INIT = (0, 0)

# DIM = (5, 4)
# GOAL_CELLS = [(1, 3), (0, 2)]
# SENSOR_RNG = 1
# P2_INIT = (4, 0)

# Define configuration
config = {
    "directory": "out",
    "filename": "4by4_rng1_fixed",
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
        return [gw_util.GW_ACT_N, gw_util.GW_ACT_E, gw_util.GW_ACT_S, gw_util.GW_ACT_W]

    def delta(self, state, act):
        p1r, p1c, p2r, p2c, turn = state

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

    @gg_models.register_property(GRAPH_PROPERTY)
    def goal_cells(self):
        return self._goal_cells


def main_multiple_init():
    # Instantiate random game here
    game = RndGridworld(dim=DIM, goal_cells=GOAL_CELLS, sense_rng=SENSOR_RNG)

    # Iterate over initial states. Fix P2's state, P1's state variable. P1 plays first.
    p2r, p2c = P2_INIT
    game_init_set = set()
    for p1r, p1c in itertools.product(range(DIM[0]), range(DIM[1])):
        # Initialize the game
        game_init_set.add((p1r, p1c, p2r, p2c, 1))

    # Run the experiment
    dirpath = config["directory"] = os.path.join("out", "multi_init_setup")
    config["filename"] = f"{pathlib.Path(__file__).name.split('.')[0]}_multi_init"
    if os.path.exists(dirpath) and os.path.isdir(dirpath):
        shutil.rmtree(dirpath)
    os.mkdir(dirpath)

    exp.run_experiment(game, config=config, game_init_set=game_init_set)


if __name__ == "__main__":
    main_multiple_init()
