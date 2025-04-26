import enum
import itertools
import sys
from typing import List, Set

import networkx as nx
import numpy as np
import spot
from loguru import logger
from prefltlf2pdfa import PrefAutomaton
from tqdm import tqdm

from ggsolver.game_tsys.constants import ModelType
from ggsolver.game_tsys.gamedef import TSys, new_state
from ggsolver.game_tsys.representations import GraphGame

logger.remove()
logger.add(sys.stderr, level="INFO")

# ==========================================================================
# GLOBAL TYPES/ENUMS
# ==========================================================================

MDPState = new_state("MDPState",
                     ["bee_x", "bee_y", "bird_x", "bird_y", "battery", "raining", "rain_prob", "terminated"])
ProdState = new_state("ProdState",
                      ["game_state", "aut_state"])


class StochasticOrderType(enum.Enum):
    WEAK = "weak"
    WEAK_STAR = "weak*"
    STRONG = "strong"


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


class ProductGame(TSys):
    def __init__(self, game: TSys | GraphGame, aut: PrefAutomaton):
        # Base constructor
        super().__init__(
            name=f"ProductGame({game.name})",
            model_type=game.model_type,
            is_qualitative=game.is_qualitative
        )

        # Save game and list of preference automaton
        self._game = game
        self._aut = aut

    def state_vars(self):
        return ProdState.components

    def states(self):
        init_states = list()
        for g_state in self._game.init_states():
            label = self._game.label(g_state)
            aut_state = self._aut.delta(self._aut.init_state, label)
            init_states.append(ProdState(g_state, aut_state))
        return init_states

    def actions(self, state: ProdState):
        return self._game.actions(state.game_state)

    def delta(self, state, action):
        next_states = dict()
        n_game_states = self._game.delta(state.game_state,
                                         action)  # For quantitative MDP, we expect dict {next_state: prob}
        for n_game_state, prob in n_game_states.items():
            n_aut_state = self._aut.delta(state.aut_state, self._game.label(n_game_state))
            next_states[ProdState(n_game_state, n_aut_state)] = prob
        return next_states

    def turn(self, state):
        return state.game_state.turn


class PrefGraphGame(GraphGame):
    def __init__(self, graph: nx.MultiDiGraph, aut: PrefAutomaton):
        super().__init__(graph)
        self._aut = aut

    @property
    def aut(self):
        return self._aut


# ==========================================================================
# SOLVERS
# ==========================================================================

class Solver:
    def __init__(self, product_mdp: PrefGraphGame, objective: List[Set[ProdState]], weight: List, **options):
        assert len(weight) == len(objective), f"Length of weight vector must be equal to that of objective."

        # Input
        self._game = product_mdp
        self._objective = objective
        self._weight = weight
        self._tol = options.get("tol", 1e-4)
        self._option_validate_mdp_stochastic = options.get("validate_mdp_stochastic", False)
        self._option_validate_policy = options.get("validate_policy", True)

        # Output
        self._matrix = None
        self._value_function = None
        self._policy = None
        self._markov_chain = None

        # Intermediate variables
        self._actions = list({data["action"] for _, _, data in self._game.edges(data=True)})
        self._act2id = {action: act_id for act_id, action in enumerate(self._actions)}
        self._state2id = {data["state"]: sid for sid, data in self._game.model.nodes(data=True)}
        self._terminal_nodes = {self._state2id[state] for state in self._game.states() if state.game_state.terminated}
        self._probability_vector = []
        self._sat_probability_of_objectives = []

    def solve(self):
        # Process objectives
        objective_nodes = [
            {self._state2id[st] for st in obj_nodes if st.game_state.terminated}
            for i, obj_nodes in enumerate(self._objective)
        ]

        # Construct reward function
        weights = np.array(self._weight)
        rewards = {
            sid: sum(np.array([1 if sid in obj_set else 0 for obj_set in objective_nodes]) * weights)
            for sid in self._terminal_nodes
        }

        # Run value iteration on MDP
        policy, values = self._value_iteration(rewards)
        self._policy = policy
        self._value_function = values

        # Run a policy check, if option is enabled.
        if self._option_validate_policy:
            self._validate_policy()

        # Marginalize MDP with policy
        mc = self._construct_markov_chain()

        # Construct probability vector for ordering classes
        self._probability_vector = self._construct_probability_vector(mc, objective_nodes)
        logger.info(f"Weight vector: {self._weight}")
        logger.info(f"Ordering classes: {objective_nodes}")
        logger.info(f"V(s0)[ordering class]: {self._value_function[0]}")
        logger.info(f"Pr(ordering class): {self._probability_vector}")

        # Compute probabilities of visiting preference graph nodes under policy
        objective_nodes = [data["partition"] for node, data in self._game.aut.pref_graph.nodes(data=True)]  # aut states
        objective_nodes = [
            {
                node
                for node, data in self._game.model.nodes(data=True)
                if (data["state"].game_state.terminated and data["state"].aut_state in o)
            }
            for o in objective_nodes
        ]
        self._sat_probability_of_objectives = self._construct_probability_vector(mc, objective_nodes)
        logger.info(f"Pref graph nodes vector: {objective_nodes}")
        logger.info(f"Pr(pi \\models phi): {self._sat_probability_of_objectives}")

    def _value_iteration(self, rewards):
        """ Using sparse arrays """
        # Construct matrix representation of MDP
        matrix = self.transition_system_to_np_array()

        # Initialize variables
        mdp_graph = self._game.model
        value_func = np.zeros((mdp_graph.number_of_nodes(), 1))
        policy = np.zeros((mdp_graph.number_of_nodes(), 1))

        # Initialize values of terminal states
        for sid, rew in rewards.items():
            value_func[sid] = rew

        # Bellman updates
        converged = False
        while not converged:
            # For each action, update Q-map (|A| x |S| x 1 matrix)
            q_func = matrix @ value_func

            # Update value func
            # n_value_func = np.maximum([mat.toarray() for mat in q_func])
            n_value_func = np.max(q_func, axis=0)
            policy = np.argmax(q_func, axis=0)

            # Check termination
            max_delta = np.abs(n_value_func - value_func).max()
            if max_delta <= self._tol:
                converged = True

            # Update value function
            value_func = n_value_func

        # Construct policy
        # policy = np.argmax(q_func_tmp, axis=0)
        policy = dict(zip(mdp_graph.nodes, [self._actions[act[0]] for act in policy.tolist()]))

        # Return
        return policy, value_func

    def transition_system_to_np_array(self):
        # MDP graph
        mdp_graph: nx.MultiDiGraph = self._game.model

        # Initialize empty transition matrix
        num_actions = len(self._actions)
        num_states = mdp_graph.number_of_nodes()
        mdp_matrix = np.zeros((num_actions, num_states, num_states))

        # Populate matrix
        for src, dst, data in mdp_graph.edges(data=True):
            act_id = self._act2id[data["action"]]
            mdp_matrix[act_id, src, dst] = data["probability"]

        # Check all rows sum up to 0 or 1
        if self._option_validate_mdp_stochastic:
            for sid in tqdm(range(num_states), desc="Checking row-stochasticity"):
                row_sums = np.sum(mdp_matrix[:, sid, :], axis=1)
                assert all(np.isclose(x, 0) or np.isclose(x, 1) for x in row_sums), \
                    f"Transition probabilities for {sid} do not sum to 0 or 1."

        return mdp_matrix

    def _construct_markov_chain(self):
        assert self._policy is not None, f"Cannot construct Markov chain when policy not computed."
        edges_to_keep = list()
        for src, dst, data in self._game.model.edges(data=True):
            if self._policy[src] == data["action"]:
                edges_to_keep.append((src, dst, data))

        mc = nx.DiGraph()
        mc.add_nodes_from(self._game.model.nodes)
        mc.add_edges_from(edges_to_keep)
        return mc
        # return nx.edge_subgraph(self._game.model, edges_to_keep)

    def _construct_probability_vector(self, mc: nx.DiGraph, objective_nodes):
        # Get S x S transition probability matrix for Markov chain.
        #   Disregard what actions are chosen, just consider probabilities.
        transition_matrix = nx.to_numpy_array(mc, weight="probability")

        # Broadcast transition probabiltiy matrix to match size of objectives (=number of nodes in pref. graph)
        n_objectives = len(objective_nodes)
        transition_matrix = np.broadcast_to(transition_matrix, (n_objectives,) + tuple(transition_matrix.shape))

        # Initialize value vector
        value_vector = np.zeros((n_objectives, transition_matrix.shape[1], 1))

        # Initialize value vector
        for i in range(len(objective_nodes)):
            for sid in objective_nodes[i]:
                # Set value vector for goal state (sid)
                value_vector[i, sid] = 1.0

        converged = False
        while not converged:
            n_value_vector = transition_matrix @ value_vector
            if (np.abs(n_value_vector - value_vector).max()) <= self._tol:
                converged = True
            value_vector = n_value_vector

        s0 = list(self._game.model.graph["init_states"])[0]
        return value_vector[:, s0].tolist()

    def _validate_policy(self):
        for src in self._game.model.nodes():
            if self._policy[src] not in self._game.actions(self._game.model.nodes[src]["state"]):
                logger.error(
                    f"Invalid policy synthesis.\n"
                    f"State {src}: {self._game.model.nodes[src]['state']} \n"
                    f"Policy: {self._policy[src]}, \n"
                    f"Available actions: {self._game.actions(self._game.model.nodes[src]['state'])}"
                )


def stochastic_weak_order(preference_graph: nx.MultiDiGraph):
    ordering = dict()
    for v in preference_graph.nodes:
        # Compute upper closure of {v}
        upper_closure_v = set(nx.descendants(preference_graph, v)) | {v}

        # Extract set of semi-automaton states belonging to a preference graph node in the upper-closure
        ordering[v] = set.union(*[set(preference_graph.nodes[node]["partition"]) for node in upper_closure_v])
    # Return ordering
    return ordering


def stochastic_weak_star_order(preference_graph: nx.MultiDiGraph):
    ordering = dict()
    for v in preference_graph.nodes:
        # Compute E \ lower_closure({v})
        lower_closure_v = set(nx.ancestors(preference_graph, v)) | {v}
        non_lower_closure_v = set(preference_graph.nodes()) - lower_closure_v

        # Extract set of semi-automaton states belonging to a preference graph node in E \ lower_closure({v})
        if len(non_lower_closure_v) == 0:
            ordering[v] = set()
        else:
            ordering[v] = set.union(
                *[set(preference_graph.nodes[node]["partition"]) for node in non_lower_closure_v])

    # Return ordering
    return ordering


def stochastic_strong_order(preference_graph: nx.MultiDiGraph):
    if preference_graph.number_of_nodes() > 8:
        raise ValueError("Computing strong stochastic order is not supported for preference graph "
                         "with 8+ nodes due to subset construction.")

    ordering = dict()
    for n in range(preference_graph.number_of_nodes() + 1):
        for subset in itertools.combinations(preference_graph.nodes, n):
            # Compute upper closure of `subset` of nodes (i.e., nodes reachable from at least one state in subset)
            upper_closure_subset = set()
            for layer in nx.bfs_layers(preference_graph, sources=subset):
                upper_closure_subset.update(layer)

            # Extract set of semi-automaton states belonging to some preference graph node in upper-closure
            if len(upper_closure_subset) == 0:
                ordering[subset] = set()
            else:
                ordering[subset] = set.union(
                    *[set(preference_graph.nodes[node]["partition"]) for node in upper_closure_subset]
                )

    # Return ordering
    return ordering


def spot_eval(cond, true_atoms):
    """
    Evaluates a propositional logic formula given the set of true atoms.

    :param true_atoms: (Iterable[str]) A propositional logic formula.
    :return: (bool) True if formula is true, otherwise False.
    """

    # Define a transform to apply to AST of spot.formula.
    def transform(node: spot.formula):
        if node.is_literal():
            if "!" not in node.to_str():
                if node.to_str() in true_atoms:
                    return spot.formula.tt()
                else:
                    return spot.formula.ff()

        return node.map(transform)

    # Apply the transform and return the result.
    # Since every literal is replaced by true or false,
    #   the transformed formula is guaranteed to be either true or false.
    return True if transform(spot.formula(cond)).is_tt() else False


# ==========================================================================
# PATCH: PrefAutomaton doesn't have delta function in-built. Adding it here.
# ==========================================================================

def aut_delta(self, q, atoms):
    # subs_map = {p: True if p in atoms else False for p in self.atoms()}
    for cond, q_next in self.transitions[q].items():
        if spot_eval(cond, atoms):
            return q_next
    raise ValueError(f"PrefAutomaton has no transition on {q=}, {atoms=}.")


PrefAutomaton.delta = aut_delta
