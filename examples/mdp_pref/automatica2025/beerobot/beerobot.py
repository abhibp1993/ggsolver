import sys

from loguru import logger
from prefltlf2pdfa import PrefAutomaton

from ggsolver import *
from ggsolver.game_tsys.gamedef import new_state
from ggsolver.mdp_prefltlf import *

logger.remove()
logger.add(sys.stderr, level="INFO")

# ==========================================================================
# GLOBAL TYPES/ENUMS
# ==========================================================================

MDPState = new_state("MDPState",
                     ["bee_x", "bee_y", "bird_x", "bird_y", "battery", "raining", "rain_prob", "terminated"])


# ==========================================================================
# MODELS
# ==========================================================================

class BeeRobotGW(TSys):
    def __init__(self, config):
        super().__init__(name="BeeRobotGW", model_type=ModelType.MDP)
        self.num_columns = config["num_columns"]
        self.num_rows = config["num_rows"]
        self.bee_initial_loc = config["bee_initial_loc"]
        self.bird_initial_loc = config["bird_initial_loc"]
        self.battery_capacity = config["battery_capacity"]
        self.bird_bounds = config["bird_bounds"]
        self.tulip_loc = config["tulip_loc"]
        self.orchid_loc = config["orchid_loc"]
        self.daisy_loc = config["daisy_loc"]
        self.bee_dynamic_stochastic = config["bee_dynamic_stochastic"]
        self.p_bee_action_success = config["bee_dynamic_stochasticity_prob"] \
            if self.bee_dynamic_stochastic else 1.0

    def state_vars(self):
        return MDPState.components

    def states(self):
        return [MDPState(*args) for args in itertools.product(
            range(self.num_columns),  # Bee robot X-position
            range(self.num_rows),  # Bee robot Y-position
            self.bird_bounds,  # Bird position
            range(self.battery_capacity + 1),  # Battery state
            [True, False],  # Raining or not raining
            range(1, 6)  # Time since last rain
            [True, False],  # terminated
        )]

    def actions(self, state):
        if state.battery == 0:
            return []

        if (state.bee_x, state.bee_y) == (state.bird_x, state.bird_y):
            return ["Y", "T"]

        possible_actions = []
        if state.bee_x < self.num_columns - 1:
            possible_actions.append('E')
        if state.bee_x > 0:
            possible_actions.append('W')
        if state.bee_y < self.num_rows - 1:
            possible_actions.append('N')
        if state.bee_y > 0:
            possible_actions.append('S')
        possible_actions.append('Y')  # Stay
        possible_actions.append('T')  # Terminate
        return possible_actions

    def delta(self, state, action):
        if state.terminated or action == 'T' or state.battery == 1:
            return {MDPState(None, None, None, None, None, None, None, True): 1.0}

        # Determine next bee location if given action is successful
        n_bee_x, n_bee_y = state.bee_x, state.bee_y
        if action == 'N':
            n_bee_y = min(state.bee_y + 1, self.num_rows - 1)
        elif action == 'S':
            n_bee_y = max(state.bee_y - 1, 0)
        elif action == 'E':
            n_bee_x = min(state.bee_x + 1, self.num_columns - 1)
        elif action == 'W':
            n_bee_x = max(state.bee_x - 1, 0)

        # Update battery state
        next_battery = state.battery - 1

        # Handle stochastic elements: bird movement and rain
        # 1. Determine potential next bird locations
        next_bird_locations = self._next_bird_locations(state)

        # 2. Determine next rain status
        if state.rain_prob == 1.0:
            # Possible rain states: (raining, next_rain_prob)-tuples
            next_rain_state = {(True, 0.2)}
        else:
            # Possible rain states: (raining, next_rain_prob)-tuples
            next_rain_state = {(True, 0.2), (False, state.rain_prob + 0.2)}

        # Construct next states
        # Transition probability calc: prob*((5-raining_status_steps)*0.2)/len(next_possible_bird_locations)
        next_states = dict()
        for (n_bird_x, n_bird_y), (n_rain, n_rain_prob) in itertools.product(
                next_bird_locations,
                next_rain_state
        ):
            # Determine next states and their probabilities based on stochasticity in bee's & bird's movement and rain
            if n_rain:
                # Successful bee action
                n_state = MDPState(n_bee_x, n_bee_y, n_bird_x, n_bird_y, next_battery, True, n_rain_prob, False)
                # Idea: n_prob = P(bee action success) * P(bird moves to chosen cell) * P(rain | state.rain_prob)
                n_prob = self.p_bee_action_success * (1 / len(next_bird_locations)) * state.rain_prob
                next_states[n_state] = next_states.get(n_state, 0.0) + n_prob

                # Bee action failed (it stays in same cell)
                n_state = MDPState(n_bee_x, n_bee_y, n_bird_x, n_bird_y, next_battery, True, n_rain_prob, False)
                # Idea: n_prob = P(bee action fails) * P(bird moves to chosen cell) * P(rain | state.rain_prob)
                n_prob = (1 - self.p_bee_action_success) * (1 / len(next_bird_locations)) * state.rain_prob
                next_states[n_state] = next_states.get(n_state, 0.0) + n_prob

            else:
                # Successful bee action
                n_state = MDPState(n_bee_x, n_bee_y, n_bird_x, n_bird_y, next_battery, False, n_rain_prob, False)
                # Idea: n_prob = P(bee action success) * P(bird moves to chosen cell) * P(rain | state.rain_prob)
                n_prob = self.p_bee_action_success * (1 / len(next_bird_locations)) * (1 - state.rain_prob)
                next_states[n_state] = next_states.get(n_state, 0.0) + n_prob

                # Failed bee action
                n_state = MDPState(n_bee_x, n_bee_y, n_bird_x, n_bird_y, next_battery, False, n_rain_prob, False)
                # Idea: n_prob = P(bee action fails) * P(bird moves to chosen cell) * P(rain | state.rain_prob)
                n_prob = (1 - self.p_bee_action_success) * (1 / len(next_bird_locations)) * (1 - state.rain_prob)
                next_states[n_state] = next_states.get(n_state, 0.0) + n_prob

        assert sum(next_states.values()) - 1 <= 1e-4, f"Sum(probs) != 1: {state=}, {action=}, {next_states=}"
        return next_states

    def init_states(self):
        # Return only initial state (we'll use pointed construction).
        return [MDPState(
            bee_x=self.bee_initial_loc[0],
            bee_y=self.bee_initial_loc[1],
            bird_x=self.bird_initial_loc[0],
            bird_y=self.bird_initial_loc[1],
            battery=self.battery_capacity,
            raining=False,
            rain_prob=0.2,
            terminated=False
        )]

    def atoms(self):
        return {"t", "o", "d"}

    def label(self, state):
        if (state.bee_x, state.bee_y) == self.tulip_loc:
            return {"t"}
        elif (state.bee_x, state.bee_y) == self.orchid_loc:
            return {"o"}
        elif (state.bee_x, state.bee_y) == self.daisy_loc:
            return {"d"}
        return set()

    def _next_bird_locations(self, state):
        """
        Compute potential next locations of the bird, ensuring they are within bird_bounds.
        """
        bird_x, bird_y = state.bird_x, state.bird_y
        potential_moves = {
            (bird_x + 1, bird_y),  # Move East
            (bird_x - 1, bird_y),  # Move West
            (bird_x, bird_y + 1),  # Move North
            (bird_x, bird_y - 1),  # Move South
            (bird_x, bird_y)  # Stay in place
        }

        # Filter moves to ensure they are within bird_bounds
        next_bird_locations = set.intersection(potential_moves, self.bird_bounds)

        return next_bird_locations


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
