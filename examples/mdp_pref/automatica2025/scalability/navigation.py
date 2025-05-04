"""
Defines parameterized navigation domain for PrefLTLf planning.

Domain description:
    * Gridworld of size (width, height)
    * Robot starts at (0, 0)
    * Robot can move in 4 directions (up, down, left, right)
    * 3 randomly chosen goal locations
    * Preferences on visiting goals is fixed
    * In each cell, there is a probability of robot dying (i.e. game terminates).
        These probabilities are assigned at random to each cell at construction time.

Reference: The navigation domain defined by ICAPS competition & used by Meilun Li's paper on MDP Preference Planning.
    https://github.com/ssanner/rddlsim/blob/master/files/final_comp/rddl/navigation_mdp.rddl
"""
import random

from ggsolver import *
from ggsolver.mdp_prefltlf import *
from prefltlf2pdfa import PrefAutomaton
from loguru import logger
import sys

logger.remove()
logger.add(sys.stderr, level="INFO")

MDPState = NewStateCls("MDPState",
                       ["x", "y", "terminated"])


class NavigationDomain(TSys):
    def __init__(self, x_max, y_max):
        # Base constructor
        super().__init__(name=f"NavigationDomain({x_max}, {y_max})", model_type=ModelType.MDP)

        # Input parameters
        self.x_max = x_max
        self.y_max = y_max

        # Gridworld parameters (generated at random)
        self._cells = list(itertools.product(range(x_max), range(y_max)))
        self._death_probability = np.random.rand(x_max, y_max)
        self._goal_locations = {f"g{idx}": cell for idx, cell in enumerate(random.sample(self._cells, k=3))}

    def state_vars(self):
        return MDPState.components

    def states(self):
        return self.init_states()

    def init_states(self):
        return [MDPState(x=0, y=0, terminated=False)]

    def actions(self, state):
        if state.terminated:
            return ["T"]

        possible_actions = []
        if state.x < self.x_max - 1:
            possible_actions.append('E')
        if state.x > 0:
            possible_actions.append('W')
        if state.y < self.y_max - 1:
            possible_actions.append('N')
        if state.y > 0:
            possible_actions.append('S')
        possible_actions.append('Y')  # Stay
        possible_actions.append('T')  # Terminate
        return possible_actions

    def delta(self, state, action):
        if state.terminated or action == 'T':
            return {MDPState(x=None, y=None, terminated=True): 1.0}

        # Determine next robot location (robot actions are deterministic)
        n_x, n_y = state.x, state.y
        if action == 'N':
            n_y = min(state.y + 1, self.y_max - 1)
        elif action == 'S':
            n_y = max(state.y - 1, 0)
        elif action == 'E':
            n_x = min(state.x + 1, self.x_max - 1)
        elif action == 'W':
            n_x = max(state.x - 1, 0)

        # Determine next state based on death probability
        death_p = self._death_probability[state.x, state.y]
        return {
            MDPState(x=None, y=None, terminated=True): death_p,
            MDPState(x=n_x, y=n_y, terminated=False): (1 - death_p)
        }

    def atoms(self):
        return {"g1", "g2", "g3"}

    def label(self, state):
        labels = set()
        for goal, loc in self._goal_locations.items():
            if (state.x, state.y) == loc:
                labels.add(goal)
        return labels


# ==========================================================================
# PATCHES
# ==========================================================================
def aut_delta(self, q, atoms):
    # subs_map = {p: True if p in atoms else False for p in self.atoms()}
    for cond, q_next in self.transitions[q].items():
        if spot_eval(cond, atoms):
            return q_next
    raise ValueError(f"PrefAutomaton has no transition on {q=}, {atoms=}.")


PrefAutomaton.delta = aut_delta
