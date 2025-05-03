from typing import List, Set

import networkx as nx
import numpy as np
from loguru import logger
from tqdm import tqdm

from ggsolver.mdp_prefltlf import *


class QuantitativePrefMDPSolver:
    def __init__(self, product_mdp: PrefGameGraph, objective: List[Set[ProdState]], weight: List, **options):
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
