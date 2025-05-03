from abc import ABC, abstractmethod

import networkx as nx

import ggsolver.game_tsys.utils as utils


class GameModel(ABC):
    """
    Abstract class to represent a game.

    This class serves as a template for various types of games and provides methods to handle game states, transitions,
    and properties such as determinism, stochasticity, concurrency, and turn-based dynamics.

    :param name: Name of the game.
    :type name: str
    :param model_type: The type of the game model (e.g., "mdp", "dtptb", "csg", "smg").
    :type model_type: ModelTypes (Enum or str)
    :param qualitative: Optional parameter to specify if the game is qualitative or quantitative.
    :type qualitative: bool, optional
    :param kwargs: Additional keyword arguments.
    """

    def __init__(self, model, **kwargs):
        """
        Initializes the Game object.

        :param name: Name of the game.
        :type name: str
        :param model_type: The type of the game model (e.g., "mdp", "dtptb", "csg", "smg").
        :type model_type: ModelTypes (Enum or str)
        :param qualitative: Indicates whether the game is qualitative (default: None).
        :type qualitative: bool, optional
        :param kwargs: Additional arguments for further customization.
        :type kwargs: dict
        """
        self._model = model

    # ----------------------------------------------------------------
    # Properties
    # ----------------------------------------------------------------
    @property
    @abstractmethod
    def name(self):
        pass

    @property
    @abstractmethod
    def model_type(self):
        pass

    @property
    def model(self):
        return self._model

    @property
    @abstractmethod
    def is_deterministic(self):
        pass

    @property
    @abstractmethod
    def is_stochastic(self):
        pass

    @property
    @abstractmethod
    def is_concurrent(self):
        pass

    @property
    @abstractmethod
    def is_turn_based(self):
        pass

    @property
    @abstractmethod
    def is_qualitative(self):
        pass

    # ----------------------------------------------------------------
    # Abstract methods: TSys API
    # ----------------------------------------------------------------
    @abstractmethod
    def states(self):
        pass

    @abstractmethod
    def init_states(self):
        pass

    @abstractmethod
    def actions(self, state):
        pass

    @abstractmethod
    def delta(self, state, action):
        pass

    @abstractmethod
    def turn(self, state):
        pass

    @abstractmethod
    def atoms(self):
        pass

    @abstractmethod
    def label(self, state):
        pass

    @abstractmethod
    def reward(self, state, action=None):
        pass


class GameGraph(GameModel):
    def __init__(self, model: nx.MultiDiGraph, **kwargs):
        """

        :param model:
        :param kwargs:
            - check_model: (Bool) Check if the model is valid. Useful if graph is constructed manually. Default: False

        """
        super().__init__(model, **kwargs)
        if kwargs.get("check_model", False):
            self._check_model()
        self._state_to_id = {data["state"]: node for node, data in model.nodes(data=True)}

    @property
    def name(self):
        return self._model.graph['name']

    @property
    def model_type(self):
        return self._model.graph['model_type']

    @property
    def is_deterministic(self):
        return utils.is_deterministic(self.model_type)

    @property
    def is_stochastic(self):
        return utils.is_stochastic(self.model_type)

    @property
    def is_concurrent(self):
        return utils.is_concurrent(self.model_type)

    @property
    def is_turn_based(self):
        return utils.is_turn_based(self.model_type)

    @property
    def is_qualitative(self):
        return self._model.graph['is_qualitative']

    # ----------------------------------------------------------------
    # Transition system API
    # ----------------------------------------------------------------
    def states(self):
        return {data["state"] for _, data in self._model.nodes(data=True)}

    def init_states(self):
        return {self._model.nodes[node]["state"] for node in self._model.graph["init_states"]}

    def actions(self, state):
        state_id = self.get_state_id(state)
        return {data["action"] for _, _, data in self._model.out_edges(state_id, data=True)}

    def delta(self, state, action):
        """
        If state is invalid, KeyError.
        If action invalid, empty dict returned -- no complaint. User is responsible to validate action.
        """
        state_id = self.get_state_id(state)
        next_states = {
            self._model.nodes[vid]["state"]: data["probability"]
            for _, vid, data in self._model.out_edges(state_id, data=True) if data["action"] == action
        }

        if self.is_deterministic:
            assert len(next_states) == 1, f"State {state} with action {action} has multiple next states: {next_states}"
            return next_states[0]
        elif self.is_stochastic and self.is_qualitative:
            return next_states.keys()
        elif self.is_stochastic and not self.is_qualitative:
            return next_states

    def turn(self, state):
        return state.turn

    def atoms(self):
        return self._model.graph["atoms"]

    def label(self, state):
        state_id = self.get_state_id(state)
        return self._model.nodes[state_id]["label"]

    def reward(self, state, action=None):
        if action is None:
            return self._model.nodes[state]["reward"]
        else:
            state_id = self.get_state_id(state)
            rewards = {
                vid: data["reward"]
                for _, vid, data in self._model.out_edges(state_id, data=True)
                if data["action"] == action
            }
            return rewards

    # ----------------------------------------------------------------
    # Graph API
    # ----------------------------------------------------------------
    def get_state_id(self, state):
        """
        Get the state ID from the state object.

        :param state: The state object.
        :return: The state ID.
        """
        return self._state_to_id[state]

    def nodes(self, data=False):
        return self._model.nodes(data=data)

    def has_node(self, node):
        return self._model.has_node(node)

    def neighbors(self, node):
        return self._model.neighbors(node)

    def successors(self, node):
        return self._model.out_neighbors(node)

    def predecessors(self, node):
        return self._model.in_neighbors(node)

    def number_of_nodes(self):
        return self._model.number_of_nodes()

    def edges(self, keys=False, data=False):
        return self._model.edges(keys=keys, data=data)

    def has_edge(self, u, v, action=None, key=None):
        if action is not None:
            return any(
                data["action"] == action
                for _, vid, data in self._model.out_edges(u, data=True)
                if vid == v
            )

        if key is not None:
            return self._model.has_edge(u, v, key=key)

        return self._model.has_edge(u, v)

    def out_edges(self, node, keys=False, data=False):
        return self._model.out_edges(node, keys=keys, data=data)

    def in_edges(self, node, keys=False, data=False):
        return self._model.in_edges(node, keys=keys, data=data)

    def number_of_edges(self):
        return self._model.number_of_edges()

    def strongly_connected_components(self):
        return nx.strongly_connected_components(self._model)

    def reachable_nodes(self, nodes: int | list[int] | set[int] | tuple[int], reverse: bool = True) -> set[int]:
        """
        Get the reachable nodes from a given node or list of nodes.

        :param reverse: If True, find predecessors; if False, find successors.
        :param nodes: The node or list of nodes to check.
        :return: A set of reachable nodes.
        """
        # Select traversal function
        next_state_func = self._model.predecessors if reverse else self._model.successors

        # Initialize queue and reachable states set
        reachable = set()
        queue_set = set(nodes)
        queue = list(nodes)

        # Run (reverse) BFS
        while queue:
            node = queue.pop(0)
            queue_set.remove(node)
            reachable.add(node)
            for vid in next_state_func(node):
                if vid not in reachable and vid not in queue_set:
                    queue_set.add(vid)
                    queue.append(vid)

        return reachable

    # ----------------------------------------------------------------
    # Helper functions
    # ----------------------------------------------------------------
    def _check_model(self, check_labels=True):
        """
        Validates the integrity of the graph model.

        Ensures that:
        - Required graph-level properties are present.
        - Each node has a 'state' attribute.
        - Each edge has 'action' and 'probability' attributes (if stochastic).
        - Probabilities on outgoing edges from a node sum to 1 (if stochastic and quantitative).
        """
        assert isinstance(self._model, nx.MultiDiGraph), "GraphGame.model must be a nx.MultiDiGraph."

        # Check required graph-level properties
        required_graph_attrs = {"name", "model_type", "is_qualitative", "atoms", "state_vars", "init_states"}
        assert required_graph_attrs.issubset(set(self._model.graph.keys())), \
            (f"Graph is missing required attribute: '{required_graph_attrs - set(self._model.graph.keys())}' or has "
             f"spurious attributes: '{set(self._model.graph.keys()) - required_graph_attrs}'.")

        # Check node attributes
        for node, data in self._model.nodes(data=True):
            if "state" not in data:
                raise ValueError(f"Node {node} is missing required attribute: 'state'.")
            if "label" not in data:
                raise ValueError(f"Node {node} is missing required attribute: 'state'.")

        # Check edge attributes
        for u, v, data in self._model.edges(data=True):
            if "action" not in data or "probability" not in data:
                raise ValueError(f"Edge ({u}, {v}) is missing required attribute: 'action' or 'probability'."
                                 f"Regardless of type of game, all edges must have 'action' AND 'probability' attribute.")

        # Check probability consistency for stochastic models
        if self.is_stochastic and not self.is_qualitative:
            for node in self._model.nodes:
                outgoing_edges = self._model.out_edges(node, data=True)
                action_probabilities = {}

                # Group probabilities by action
                for _, _, data in outgoing_edges:
                    action = data["action"]
                    probability = data["probability"]
                    if action not in action_probabilities:
                        action_probabilities[action] = []
                    action_probabilities[action].append(probability)

                # Check if probabilities for each action sum to 1
                for action, probabilities in action_probabilities.items():
                    if probabilities and not abs(sum(probabilities) - 1.0) < 1e-6:
                        raise ValueError(
                            f"Probabilities of outgoing edges with action '{action}' from node {node} do not sum to 1.")

