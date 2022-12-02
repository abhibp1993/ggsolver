import inspect
import itertools
import logging
import random
import typing
from functools import partial
from ggsolver import util
from ggsolver.graph import NodePropertyMap, EdgePropertyMap, Graph, SubGraph
from tqdm import tqdm

# try:
#     import ggsolver.logic.pl as pl
# except ImportError as err:
#     import traceback
#     logging.error(util.ColoredMsg.error(f"[ERROR] logic.pl could not be loaded. Logic functionality will not work. "
#                                         f"\nError: {err.with_traceback(None)}"))


# ==========================================================================
# DECORATOR FUNCTIONS.
# ==========================================================================
def register_property(property_set: set):
    def register_function(func):
        if func.__name__ in property_set:
            print(util.BColors.WARNING, f"[WARN] Duplicate property: {func.__name__}.", util.BColors.ENDC)
        property_set.add(func.__name__)
        return func
    return register_function


# ==========================================================================
# BASE CLASS.
# ==========================================================================
class GraphicalModel:
    NODE_PROPERTY = set()
    EDGE_PROPERTY = set()
    GRAPH_PROPERTY = set()

    def __init__(self, is_deterministic=True, is_probabilistic=False, **kwargs):
        # Types of Graphical Models. Acceptable values:
        self._is_deterministic = is_deterministic
        self._is_probabilistic = is_probabilistic

        # Input domain (Expected value: Name of the function that returns an Iterable object.)
        self._input_domain = kwargs["input_domain"] if "input_domain" in kwargs else None

        # Pointed model
        self._init_state = kwargs["init_state"] if "init_state" in kwargs else None

        # Caching variables during serializing and deserializing the model.
        self.__graph = None
        self.__is_graphified = False
        self.__states = list()
        self.__state2node = dict()

    def __str__(self):
        return f"<{self.__class__.__name__} object at {id(self)}>"

    def __setattr__(self, key, value):
        # If key is any non "__xxx" variable, set `is_graphified` to False.
        if key != "__is_graphified" and hasattr(self, "__is_graphified"):
            if key[0:2] != "__":
                self.__is_graphified = False
        super(GraphicalModel, self).__setattr__(key, value)

    # ==========================================================================
    # PRIVATE FUNCTIONS.
    # ==========================================================================
    def _clear_cache(self):
        self.__states = dict()
        self.__is_graphified = False

    def _gen_edges(self, delta, state, inp):
        next_states = delta(state, inp)
        edges = set()

        # There are three types of graphical models. Handle each separately.
        # If model is deterministic, next states is a single state.
        if self.is_deterministic():
            if next_states is None:
                logging.warning(
                    util.ColoredMsg.warn(f"[WARN] {self.__class__.__name__}._graphify_unpointed(): "
                                         f"No edge(s) added to graph for state={state}, input={inp}, "
                                         f"next_state={next_states}.")
                )
                return set()

            edges.add((state, next_states, inp, None))

        # If model is non-deterministic, next states is an Iterable of states.
        elif not self.is_deterministic() and not self.is_probabilistic():
            for next_state in next_states:
                if next_state is None:
                    logging.warning(
                        util.ColoredMsg.warn(f"[WARN] {self.__class__.__name__}._graphify_unpointed(): "
                                             f"No edge(s) added to graph for state={state}, input={inp}, "
                                             f"next_state={next_state}.")
                    )
                    continue

                edges.add((state, next_state, inp, None))

        # If model is stochastic, next states is a Distribution of states.
        elif not self.is_deterministic() and self.is_probabilistic():
            # FIXME. I have doubts that following implementation is correct.
            #  If support is empty, the code under `if` will not execute.
            for next_state in next_states.support():
                if next_state is None:
                    logging.warning(
                        util.ColoredMsg.warn(f"[WARN] {self.__class__.__name__}._graphify_unpointed(): "
                                             f"No edge(s) added to graph for state={state}, input={inp}, "
                                             f"next_state={next_state}.")
                    )
                    continue

                edges.add((state, next_state, inp, next_states.pmf(next_state)))

        else:
            raise TypeError("Graphical Model is neither deterministic, nor non-deterministic, nor stochastic! "
                            f"Check the values: is_deterministic: {self.is_deterministic()}, "
                            f"self.is_quantitative:{self.is_probabilistic()}.")

        return edges

    def _gen_underlying_graph_unpointed(self, graph):
        """
        Programmer's notes:
        1. Caches states (returned by `self.states()`) in self.__states variable.
        2. Assumes all states to be hashable.
        3. (in v0.1.7) Handles `enabled_acts` optional function.
        """
        # Get states
        states = getattr(self, "states")
        states = list(states())

        # Add states to graph
        node_ids = list(graph.add_nodes(len(states)))

        # Cache states as a dictionary {state: uid}
        self.__states = dict(zip(states, node_ids))

        # Node property: state
        np_state = NodePropertyMap(graph=graph)
        np_state.update(dict(zip(node_ids, states)))
        graph["state"] = np_state

        # Logging and printing
        logging.info(util.ColoredMsg.ok(f"[INFO] Processed node property: states. Added {len(node_ids)} states. [OK]"))

        # Get input function
        input_func = getattr(self, self._input_domain)
        logging.info(util.ColoredMsg.ok(f"[INFO] Input domain function detected as '{self._input_domain}'. [OK]"))

        # Graph property: input domain (stores the name of edge property that represents inputs)
        graph["input_domain"] = self._input_domain
        logging.info(util.ColoredMsg.ok(f"[INFO] Processed graph property: input_domain. [OK]"))

        # Get input domain
        inputs = list(input_func())

        # Edge properties: input, prob,
        ep_input = EdgePropertyMap(graph=graph)
        ep_prob = EdgePropertyMap(graph=graph, default=None)

        # Process enabled actions function, if provided.
        enabled_acts = getattr(self, "enabled_acts", None)
        if enabled_acts is not None:
            try:
                # Check if enabled_acts is implemented by user.
                arbitrary_key, st = list(self.__states.items())[0]
                enabled_acts(st)
            except NotImplementedError:
                enabled_acts = None

        # If enabled_acts is defined and implemented, then define and store the property.
        if enabled_acts is not None:
            np_enabled_acts = NodePropertyMap(graph=graph, default=inputs)
            for state in tqdm(self.__states.keys(), desc="Generating state: enabled actions map"):
                np_enabled_acts[self.__states[state]] = enabled_acts(state)
            graph["enabled_acts"] = np_enabled_acts
            logging.info(util.ColoredMsg.ok(f"[INFO] Processed node property: enabled_acts. [OK]"))

        # Generate edges
        delta = getattr(self, "delta")

        # for state, inp in tqdm(itertools.product(self.__states.keys(), inputs),
        #                        total=len(self.__states) * len(inputs),
        #                        desc="Unpointed graphify adding edges"):

        for state in tqdm(self.__states.keys(), desc="Unpointed graphify adding edges"):
            # Get the enabled inputs at the state. If enabled_acts is not defined, then use entire inputs set.
            if enabled_acts is not None:
                inputs_at_state = np_enabled_acts[self.__states[state]]
            else:
                inputs_at_state = inputs

            # Apply inputs to state to generate out edges
            for inp in inputs_at_state:
                # Generate edges from state
                new_edges = self._gen_edges(delta, state, inp)

                # Update graph edges
                uid = self.__states[state]
                for _, t, _, prob in new_edges:
                    vid = self.__states[t]
                    key = graph.add_edge(uid, vid)
                    ep_input[uid, vid, key] = inp
                    ep_prob[uid, vid, key] = prob

        # Add edge properties to graph
        graph["input"] = ep_input
        graph["prob"] = ep_prob
        logging.info(util.ColoredMsg.ok(f"[INFO] Processed edge property: input. [OK]"))
        logging.info(util.ColoredMsg.ok(f"[INFO] Processed graph property: prob. [OK]"))

    def _gen_underlying_graph_pointed(self, graph):
        logging.info(util.ColoredMsg.ok(f"[INFO] Running graphify UNPOINTED."))

        # Get input function
        input_func = getattr(self, self._input_domain)
        logging.info(util.ColoredMsg.ok(f"[INFO] Input domain function detected as '{self._input_domain}'. [OK]"))

        # Graph property: input domain (stores the name of edge property that represents inputs)
        graph["input_domain"] = self._input_domain
        logging.info(util.ColoredMsg.ok(f"[INFO] Processed graph property: input_domain. [OK]"))

        # Get input domain
        inputs = list(input_func())

        # Node property: state
        np_state = NodePropertyMap(graph=graph)

        # Edge properties: input, prob,
        ep_input = EdgePropertyMap(graph=graph)
        ep_prob = EdgePropertyMap(graph=graph, default=None)

        # Get initial state
        s0 = self.init_state()

        # Process enabled actions function, if provided.
        enabled_acts = getattr(self, "enabled_acts", None)
        if enabled_acts is not None:
            try:
                # Check if enabled_acts is implemented by user.
                enabled_acts(s0)
            except NotImplementedError:
                enabled_acts = None

        # If enabled_acts is defined and implemented, then define and store the property.
        if enabled_acts is not None:
            np_enabled_acts = NodePropertyMap(graph=graph, default=inputs)
            # for state in tqdm(self.__states.keys(), desc="Generating state: enabled actions map"):
            #     np_enabled_acts[state] = enabled_acts(state)
            # graph["enabled_acts"] = np_enabled_acts
            # logging.info(util.ColoredMsg.ok(f"[INFO] Processed node property: enabled_acts. [OK]"))

        # BFS traversal until all reachable states are visited.
        uid = graph.add_node()
        self.__states[s0] = uid
        np_state[uid] = s0

        queue = [s0]
        visited = set()

        # Generate edges
        delta = getattr(self, "delta")
        with tqdm(total=1, desc="Pointed graphify adding edges") as progress_bar:
            while len(queue) > 0:
                # Update progress_bar
                progress_bar.total = len(queue) + len(visited)
                progress_bar.update(1)

                # Visit a state. Add to graph. Update cache. Update node property `state`.
                state = queue.pop()
                visited.add(state)
                uid = self.__states[state]
                np_state[uid] = state

                # Apply all inputs to state
                if enabled_acts is not None:
                    inputs_at_state = enabled_acts(state)
                    np_enabled_acts[uid] = inputs_at_state
                else:
                    inputs_at_state = inputs

                for inp in inputs_at_state:
                    # Get successors: set of (from_st, to_st, inp, prob)
                    new_edges = self._gen_edges(delta, state, inp)

                    for _, to_state, inp, prob in new_edges:
                        # If to_state was added to queue in the past, its id will be cached.
                        # Otherwise, add new node, cache it and queue it for exploration.
                        if to_state in self.__states:
                            vid = self.__states[to_state]
                        else:
                            vid = graph.add_node()
                            self.__states[to_state] = vid
                            # np_state[uid] = state
                            queue.append(to_state)

                        # Add edge to graph
                        key = graph.add_edge(uid, vid)

                        # Set edge properties
                        ep_input[uid, vid, key] = inp
                        ep_prob[uid, vid, key] = prob

        # Add node properties to graph
        graph["state"] = np_state
        if enabled_acts is not None:
            graph["enabled_acts"] = np_enabled_acts

        # Add edge properties to graph
        graph["input"] = ep_input
        graph["prob"] = ep_prob
        logging.info(util.ColoredMsg.ok(f"[INFO] Processed edge property: input. [OK]"))
        logging.info(util.ColoredMsg.ok(f"[INFO] Processed graph property: prob. [OK]"))

    def _add_node_prop_to_graph(self, graph, p_name, default=None):
        """
        Adds the node property called `p_name` to the graph.

        Requires: `p_name` should be a function in self that inputs a single parameter: state.

        Assumes: self._add_nodes_to_graph() is called before.
        """
        if graph.has_property(p_name):
            logging.warning(util.ColoredMsg.warn(f"[WARN] Duplicate property is ignored: {p_name}. [IGNORED]"))
            return

        try:
            p_map = NodePropertyMap(graph=graph, default=default)
            p_func = getattr(self, p_name)   # self.NODE_PROPERTY[p_name]
            if not (inspect.isfunction(p_func) or inspect.ismethod(p_func)):
                raise TypeError(f"Node property {p_func} is not a function.")
            # for uid in range(len(self.__states)):
            #     p_map[uid] = p_func(self.__states[uid])
            #
            for uid in range(graph.number_of_nodes()):
                p_map[uid] = p_func(graph["state"][uid])
            graph[p_name] = p_map
            logging.info(util.ColoredMsg.ok(f"[INFO] Processed node property: {p_name}. [OK]"))
        except NotImplementedError:
            logging.warning(util.ColoredMsg.warn(f"[WARN] Node property function not implemented: {p_name}. [IGNORED]"))
        except AttributeError:
            logging.warning(util.ColoredMsg.warn(f"[WARN] Node property function is not defined: {p_name}. [IGNORED]"))

    def _add_edge_prop_to_graph(self, graph, p_name, default=None):
        """
        Adds an edge property called `p_name` to the graph.

        Requires: `p_name` should be a function in self that inputs three parameters: state, inp, next_state .

        Assumes: self._add_nodes_to_graph() is called.
        Assumes: self._add_edges_to_graph() is called.
        """
        if graph.has_property(p_name):
            logging.warning(util.ColoredMsg.warn(f"[WARN] Duplicate property: {p_name}. [IGNORED]"))
            return

        try:
            p_map = EdgePropertyMap(graph=graph, default=default)
            p_func = getattr(self, p_name)
            if not (inspect.isfunction(p_func) or inspect.ismethod(p_func)):
                raise TypeError(f"Edge property {p_func} is not a function.")
            for uid, vid, key in graph.edges():
                p_map[(uid, vid, key)] = p_func(graph["states"][uid],
                                                graph["input"][(uid, vid, key)],
                                                graph["states"][vid])
            graph[p_name] = p_map
            logging.info(util.ColoredMsg.ok(f"[INFO] Processed edge property: {p_name}. [OK]"))
        except NotImplementedError:
            logging.warning(util.ColoredMsg.warn(f"[WARN] Edge property not implemented: {p_name}. [IGNORED]"))
        except AttributeError:
            logging.warning(util.ColoredMsg.warn(f"[WARN] Node property function is not defined: {p_name}. [IGNORED]"))

    def _add_graph_prop_to_graph(self, graph, p_name):
        """
        Adds a graph property called `p_name` to the graph.

        Requires: `p_name` should be a function in self that inputs no parameters or a non-callable value.
        """
        if graph.has_property(p_name):
            logging.warning(util.ColoredMsg.warn(f"[WARN] Duplicate property: {p_name}. [IGNORED]"))
            return

        try:
            p_func = getattr(self, p_name)
            if inspect.ismethod(p_func) or (inspect.isfunction(p_func) and p_func.__name__ == "<lambda>"):
                graph[p_name] = p_func()
                logging.info(util.ColoredMsg.ok(f"[INFO] Processed graph property: {p_name}. [OK]"))
            elif inspect.isfunction(p_func):
                if len(inspect.signature(p_func).parameters) == 0:
                    graph[p_name] = p_func()
                else:
                    graph[p_name] = p_func(self)
                logging.info(util.ColoredMsg.ok(f"[INFO] Processed graph property: {p_name}. [OK]"))
            # elif inspect.ismethod(p_func):
            #     graph[p_name] = p_func()
            #     logging.warning(util.ColoredMsg.ok(f"[INFO] Processed graph property: {p_name}. OK"))
            else:
                raise TypeError(f"Graph property {p_name} is neither a function nor a method.")
        except NotImplementedError:
            logging.warning(util.ColoredMsg.warn(f"[WARN] Graph property is not implemented: {p_name}. [IGNORED]"))
        except AttributeError:
            logging.warning(util.ColoredMsg.warn(f"[WARN] Node property function is not defined: {p_name}. [IGNORED]"))

    # ==========================================================================
    # FUNCTIONS TO BE IMPLEMENTED BY USER.
    # ==========================================================================
    def states(self):
        """
        Defines the states component of the graphical model.

        :return: (list/tuple of JSONifiable object). List or tuple of states.
        """
        raise NotImplementedError(f"{self.__class__.__name__}.states() is not implemented.")

    def delta(self, state, inp) -> typing.Union[util.Distribution, typing.Iterable, object]:
        pass

    # ==========================================================================
    # PUBLIC FUNCTIONS.
    # ==========================================================================
    def initialize(self, state):
        """
        Sets the initial state of the graphical model.

        .. note:: The function does NOT check if the given state is valid.
        """
        self._init_state = state

    def graphify(self, pointed=False, base_only=False):
        """
        Constructs the underlying graph of the graphical model.

        :param pointed: (bool) If pointed is `True`, the :py:meth:`TSys.graphify_pointed()` is called, which constructs
            a pointed graphical model containing only the states reachable from the initial state.  Otherwise,
            :py:meth:`TSys.graphify_unpointed()` is called, which constructs the complete transition system.
        :return: (:class:`ggsolver.graph.Graph` object) An equivalent graph representation of the graphical model.
        """
        # Clear cached information
        self._clear_cache()

        # Input parameter validation
        if pointed is True and self._init_state is None:
            raise ValueError(f"{self.__class__.__name__} is not initialized. "
                             f"Did you forget to call {self.__class__.__name__}.initialize() function?")

        # Initialize graph object
        graph = Graph()

        # Glob node, edge and graph properties
        node_props = getattr(self, "NODE_PROPERTY")
        edge_props = getattr(self, "EDGE_PROPERTY")
        graph_props = getattr(self, "GRAPH_PROPERTY")

        # Warn about duplication
        logging.info(util.ColoredMsg.header(f"[INFO] Globbed node properties: {node_props}"))
        logging.info(util.ColoredMsg.header(f"[INFO] Globbed edge properties: {edge_props}"))
        logging.info(util.ColoredMsg.header(f"[INFO] Globbed graph properties: {graph_props}"))
        logging.info(util.ColoredMsg.header(f"[INFO] Duplicate node, edge properties: "
                                            f"{set.intersection(node_props, edge_props)}"))
        logging.info(util.ColoredMsg.header(f"[INFO] Duplicate edge, graph properties: "
                                            f"{set.intersection(edge_props, graph_props)}"))
        logging.info(util.ColoredMsg.header(f"[INFO] Duplicate graph, node properties: "
                                            f"{set.intersection(graph_props, node_props)}"))

        # Construct underlying graph for pointed construction
        if pointed is True:
            self._gen_underlying_graph_pointed(graph)

        # Construct underlying graph for unpointed construction
        else:
            self._gen_underlying_graph_unpointed(graph)

        if not base_only:
            # Add node properties
            for p_name in node_props:
                self._add_node_prop_to_graph(graph, p_name)

            # Add edge properties
            for p_name in edge_props:
                self._add_edge_prop_to_graph(graph, p_name)

            # Add graph properties
            for p_name in graph_props:
                self._add_graph_prop_to_graph(graph, p_name)
        else:
            print(util.BColors.WARNING, f"[WARN] Ignoring node, edge and graph (base_only: True)", util.BColors.ENDC)

        print(util.BColors.OKGREEN, f"[SUCCESS] {graph} generated.", util.BColors.ENDC)
        return graph

    def serialize(self):
        """
        Serializes the underlying graph of the graphical model into a dictionary with the following format.
        The state properties are saved as node properties, transition properties are stored are edge properties
        and model properties are stored as graph properties in the underlying graph::

            {
                "graph": {
                    "nodes": <number of nodes>,
                    "edges": {
                        uid: {vid: key},
                        ...
                    }
                    "node_properties": {
                        "property_name": {
                            "default": <value>,
                            "dict": {
                                "uid": <property value>,
                                ...
                            }
                        },
                        ...
                    },
                    "edge_properties": {
                        "property_name": {
                            "default": <value>,
                            "dict": [{"edge": [uid, vid, key], "pvalue": <property value>} ...]
                        },
                        ...
                    },
                    "graph_properties": {
                        "property_name": <value>,
                        ...
                    }
                }
            }

        :return: (dict) Serialized graphical model.
        """
        # 1. Graphify
        # 2. Serialize the graph
        # 3. Return a dict
        raise NotImplementedError

    def save(self, fpath, pointed=False, overwrite=False, protocol="json"):
        """
        Saves the graphical model to file.

        :param fpath: (str) Path to which the file should be saved. Must include an extension.
        :param pointed: (bool) If pointed is `True`, the :py:meth:`TSys.graphify_pointed()` is called, which constructs
            a pointed graphical model containing only the states reachable from the initial state.  Otherwise,
            :py:meth:`TSys.graphify_unpointed()` is called, which constructs the complete transition system.
        :param overwrite: (bool) Specifies whether to overwrite the file, if it exists. [Default: False]
        :param protocol: (str) The protocol to use to save the file. Options: {"json" [Default], "pickle"}.

        .. note:: Pickle protocol is not tested.
        """
        # 1. Graphify
        graph = self.graphify(pointed=pointed)

        # 2. Save the graph
        graph.save(fpath, overwrite=overwrite, protocol=protocol)

    @classmethod
    def deserialize(cls, obj_dict):
        """
        Constructs a graphical model from a serialized graph object. The node properties are deserialized as state
        properties, the edge properties are deserialized as transition properties, and the graph properties are
        deserialized as model properties. All the deserialized properties are represented as a function in the
        GraphicalModel class. See example #todo.

        The format is described in :py:meth:`GraphicalModel.serialize`.

        :return: (Sub-class of GraphicalModel) An instance of the `cls` class. `cls` must be a sub-class of
            `GraphicalModel`.
        """
        # 1. Construct a graph from obj_dict.
        # 2. Define functions from graph
        # 3. Create cls() instance.
        # 4. Update __dir__ with new methods
        # 5. Return instance
        raise NotImplementedError

    @classmethod
    def load(cls, fpath, protocol="json"):
        """
        Loads the graphical model from file.

        :param fpath: (str) Path to which the file should be saved. Must include an extension.
        :param protocol: (str) The protocol to use to save the file. Options: {"json" [Default], "pickle"}.

        .. note:: Pickle protocol is not tested.
        """
        # Load game graph
        graph = Graph.load(fpath, protocol=protocol)

        # Create object
        obj = cls()

        # Add graph properties
        for gprop, gprop_value in graph.graph_properties.items():
            func_code = f"""def {gprop}():\n\treturn {gprop_value}"""
            exec(func_code)
            func = locals()[gprop]
            setattr(obj, gprop, func)

        # Construct inverse state mapping
        for node in graph.nodes():
            state = graph["state"][node]
            if isinstance(state, list):
                state = tuple(state)
            obj.__state2node[state] = node

        # Add node properties
        def get_node_property(state, name):
            return graph.node_properties[name][obj.__state2node[state]]

        for nprop, nprop_value in graph.node_properties.items():
            setattr(obj, nprop, partial(get_node_property, name=nprop))

        # TODO. Add edge properties (How to handle them is unclear).

        # Reconstruct delta function
        def delta(state, act):
            # Get node from state
            node = obj.__state2node[state]

            # Get out_edges from node in graph
            out_edges = graph.out_edges(node)

            # Iterate over each out edge to match action.
            successors = set()
            for uid, vid, key in out_edges:
                action_label = graph["act"][(uid, vid, key)]
                if action_label == act:
                    successors.add(vid)

            # If model is deterministic, then return single state.
            if not graph["is_stochastic"]:
                return graph["state"][successors.pop()]

            # If model is stochastic and NOT quantitative, then return list of states.
            elif graph["is_stochastic"] and not graph["is_quantitative"]:
                return [graph["state"][vid] for vid in successors]

            # If model is stochastic and quantitative, then return distribution.
            else:
                successors = [graph["state"][vid] for vid in successors]
                prob = [graph["prob"][uid] for uid in successors]
                return util.Distribution(successors, prob)

        obj.delta = delta

        # Return reconstructed object
        return obj

    @classmethod
    def from_graph(cls, graph):
        # Create object
        obj = cls()

        # Save graph
        # TODO. This is inefficent because we are storing `__states` and `__graph`. Remove redundancy.
        obj.__graph = graph
        obj.__is_graphified = True

        # Add graph properties
        for gprop, gprop_value in graph.graph_properties.items():
            # func_code = f"""def {gprop}():\n\treturn {gprop_value}"""
            func = lambda: obj.__graph.graph_properties[gprop]
            # exec(func_code)
            # func = locals()[gprop]
            setattr(obj, gprop, func)

        # Construct inverse state mapping
        for node in graph.nodes():
            state = graph["state"][node]
            if isinstance(state, list):
                state = tuple(state)
            obj.__state2node[state] = node

        # Add node properties
        def get_node_property(state, name):
            return graph.node_properties[name][obj.__state2node[state]]

        for nprop, nprop_value in graph.node_properties.items():
            setattr(obj, nprop, partial(get_node_property, name=nprop))

        # TODO. Add edge properties (How to handle them is unclear).

        # Reconstruct delta function
        def delta(state, act):
            # Get node from state
            node = obj.__state2node[state]

            # Get out_edges from node in graph
            out_edges = graph.out_edges(node)

            # Iterate over each out edge to match action.
            successors = set()
            for uid, vid, key in out_edges:
                action_label = graph["act"][(uid, vid, key)]
                if action_label == act:
                    successors.add(vid)

            # If model is deterministic, then return single state.
            if not graph["is_stochastic"]:
                return graph["state"][successors.pop()]

            # If model is stochastic and NOT quantitative, then return list of states.
            elif graph["is_stochastic"] and not graph["is_quantitative"]:
                return [graph["state"][vid] for vid in successors]

            # If model is stochastic and quantitative, then return distribution.
            else:
                successors = [graph["state"][vid] for vid in successors]
                prob = [graph["prob"][uid] for uid in successors]
                return util.Distribution(successors, prob)

        obj.delta = delta

        # Return reconstructed object
        return obj

    def is_nondeterministic(self):
        """ Returns `True` if the graphical model is non-deterministic. Else, returns `False`. """
        return not self._is_deterministic and not self._is_probabilistic

    @register_property(GRAPH_PROPERTY)
    def init_state(self):
        """
        Returns the initial state of the graphical model.
        """
        return self._init_state

    @register_property(GRAPH_PROPERTY)
    def is_deterministic(self):
        """
        Returns `True` if the graphical model is deterministic. Else, returns `False`.
        """
        return self._is_deterministic

    @register_property(GRAPH_PROPERTY)
    def is_probabilistic(self):
        """ Returns `True` if the graphical model is probabilistic. Else, returns `False`. """
        return self._is_probabilistic


# ==========================================================================
# USER MODELS.
# ==========================================================================
class TSys(GraphicalModel):
    """
    Represents a transition system [Principles of Model Checking, Def. 2.1].

    .. math::
        TSys = (S, A, T, AP, L)


    In the `TSys` class, each component is represented as a function.

    - The set of states :math:`S` is represented by `TSys.states` function,
    - The set of actions :math:`A` is represented by `TSys.actions` function,
    - The transition function :math:`T` is represented by `TSys.delta` function,
    - The set of atomic propositions is represented by `TSys.atoms` function,
    - The labeling function :math:`L` is represented by `TSys.label` function.

    All of the above functions are marked abstract.
    The recommended way to use `TSys` class is by subclassing it and implementing its component functions.

    A transition system can be either deterministic or non-deterministic or probabilistic.
    To define a **deterministic** transition system, provide a keyword argument `is_deterministic=True`
    to the constructor. To define a **nondeterministic** transition system, provide a keyword argument
    `is_deterministic=False` to the constructor. To define a **probabilistic** transition system, provide
    a keyword arguments `is_deterministic=False, is_probabilistic=True` to the constructor.

    The design of `TSys` class closely follows its mathematical definition.
    Hence, the signatures of `delta` function for deterministic, nondeterministic, probabilistic
    transition systems are different.

    - **deterministic:**  `delta(state, act) -> single state`
    - **non-deterministic:**  `delta(state, act) -> a list of states`
    - **probabilistic:**  `delta(state, act) -> a distribution over states`

    An important feature of `TSys` class is the `graphify()` function. It constructs a `Graph` object that is equivalent to the transition system. The nodes of the `Graph` represent the states of `TSys`, the edges of the `Graph` are defined by the set of `actions` and the `delta` function. The atomic propositions, labeling function are stored as `node, edge` and `graph` properties. By default, every `Graph` returned a `TSys.graphify()` function have the following (node/edge/graph) properties:

    - `state`: (node property) A Map from every node to the state of transition system it represents.
    - `actions`: (graph property) List of valid actions.
    - `input`: (edge property) A map from every edge `(uid, vid, key)` to its associated action label.
    - `prob`: (edge property) The probability associated with the edge `(uid, vid, key)`.
    - `atoms`: (graph property) List of valid atomic propositions.
    - `label`: (node property) A map every node to the list of atomic propositions true in the state represented by that node.
    - `init_state`: (graph property) Initial state of transition system.
    - `is_deterministic`: (graph property) Is the transition system deterministic?
    - `is_probabilistic`: (graph property) Is the transition system probabilistic?

    **Note:** Some features of probabilistic transition system are not tested. If you are trying to implement a probabilistic transition system, reach out to Abhishek Kulkarni (a.kulkarni2@ufl.edu).


    Next, we demonstrate how to use `TSys` class to define a deterministic, non-deterministic and probabilistic transition system.

    """
    NODE_PROPERTY = GraphicalModel.NODE_PROPERTY.copy()
    EDGE_PROPERTY = GraphicalModel.EDGE_PROPERTY.copy()
    GRAPH_PROPERTY = GraphicalModel.GRAPH_PROPERTY.copy()

    def __init__(self, is_deterministic=True, is_probabilistic=False, **kwargs):
        """
        Constructs a transition system.

        :param is_deterministic: (bool). If `True` then the transition system is deterministic. Otherwise,
            it is either non-deterministic or probabilistic.
        :param is_probabilistic: (bool). If `is_deterministic` is `False`, then if `is_probabilistic` is `True`
            then the transition system is probabilistic. Otherwise, it is non-deterministic.
        :param input_domain: (optional, str). Name of the member function of TSys class that defines the inputs to the
            transition system. [Default: "actions"]
        :param init_state: (optional, JSON-serializable object). The initial state of the transition system.
        """
        kwargs["input_domain"] = "actions" if "input_domain" not in kwargs else kwargs["input_domain"]
        kwargs["is_deterministic"] = is_deterministic
        kwargs["is_probabilistic"] = is_probabilistic
        super(TSys, self).__init__(**kwargs)

    # ==========================================================================
    # FUNCTIONS TO BE IMPLEMENTED BY USER.
    # ==========================================================================
    @register_property(GRAPH_PROPERTY)
    def actions(self):
        """
        Defines the actions component of the transition system.

        :return: (list/tuple of str). List or tuple of action labels.
        """
        raise NotImplementedError(f"{self.__class__.__name__}.actions() is not implemented.")

    def delta(self, state, act):
        """
        Defines the transition function of the transition system.

        :param state: (object) A valid state.
        :param act: (str) An action.
        :return: (object/list(object)/util.Distribution object). Depending on the type of transition system, the return
            type is different.
            - Deterministic: returns a single state.
            - Non-deterministic: returns a list/tuple of states.
            - Probabilistic: returns a distribution over all state.
        """
        raise NotImplementedError(f"{self.__class__.__name__}.delta() is not implemented.")

    @register_property(GRAPH_PROPERTY)
    def atoms(self):
        """
        Defines the atomic propositions component of the transition system.

        :return: (list/tuple of str). List or tuple of atomic proposition labels.
        """
        raise NotImplementedError(f"{self.__class__.__name__}.atoms() is not implemented.")

    @register_property(NODE_PROPERTY)
    def label(self, state):
        """
        Defines the labeling function of the transition system.

        :param state: (object) A valid state.
        :return: (list/tuple of str). A list/tuple of atomic propositions that are true in the given state.
        """
        raise NotImplementedError(f"{self.__class__.__name__}.label() is not implemented.")

    @register_property(NODE_PROPERTY)
    def enabled_acts(self, state):
        """
        Defines the enabled actions at the given state.

        :param state: (object) A valid state.
        :return: (list/tuple of str). A list/tuple of actions enabled in the given state.
        """
        raise NotImplementedError(f"{self.__class__.__name__}.enabled_acts() is not implemented.")


class Game(TSys):
    """
    Represents a game transition system, hereafter referred simply as a game.
    A `Game` can represent two mathematical structures commonly used in literature, namely

    .. math::
        G = (S, A, T, AP, L, formula)

    .. math::
        G = (S, A, T, F, WinCond)

    In the `Game` class, each component is represented as a function. By defining the relevant functions, a `Game`
    class may represent either of the two mathematical structures.

    - The set of states :math:`S` is represented by `Game.states` function,
    - The set of actions :math:`A` is represented by `Game.actions` function,
    - The transition function :math:`T` is represented by `Game.delta` function,
    - The set of atomic propositions :math:`AP` is represented by `Game.atoms` function,
    - The labeling function :math:`L` is represented by `Game.label` function.
    - When the winning condition is represented by a logic formula :math:`formula`, we define `Game.formula` function.
    - When the winning condition is represented by a final states :math:`F`, we define `Game.final` function.
      In this case, we must also specify the acceptance condition.
    - The winning condition :math:`WinCond` is represented by `Game.win_cond` function.

    All of the above functions are marked abstract.
    The recommended way to use `Game` class is by subclassing it and implementing the relevant component functions.

    **Categorization of a Game:** A game is categorized by three types:

    -   A game can be either deterministic or non-deterministic or probabilistic.
        To define a **deterministic** transition system, provide a keyword argument `is_deterministic=True` to the
        constructor. To define a **nondeterministic** transition system, provide a keyword argument `is_deterministic=False`
        to the constructor. To define a **probabilistic** transition system, provide a keyword arguments
        `is_deterministic=False, is_probabilistic=True` to the constructor.

        The design of `Game` class closely follows its mathematical definition.
        Hence, the signatures of `delta` function for deterministic, nondeterministic, probabilistic games are different.

        - **deterministic:**  `delta(state, act) -> single state`
        - **non-deterministic:**  `delta(state, act) -> a list of states`
        - **probabilistic:**  `delta(state, act) -> a distribution over states`

    -   A game can be turn-based or concurrent. To define a **concurrent** game, provide a keyword argument
        `is_turn_based=False`. The game is `turn_based` by default.

    -   A game can be a 1/1.5/2/2.5-player game. A one-player game models a deterministic motion planning-type problem in
        a static environment. A 1.5-player game is an MDP. A two-player game models a deterministic interaction between
        two strategic players. And, a 2.5-player game models a stochastic interaction between two strategic players.

        If a game is one or two player, then the :py:meth:`Game.delta` is `deterministic`.
        If a game is 1.5 or 2.5 player, then the :py:meth:`Game.delta` is either `non-deterministic` (when
        transition probabilities are unknown), and `probabilistic` (when transition probabilities are known).

    Every state in a turn-based game is controlled by a player. To define which player controls which state, define
    a game component :py:meth:`Game.turn` which takes in a state and returns a value between 0 and 3 to indicate
    which player controls the state.

    An important feature of `Game` class is the `graphify()` function. It constructs a `Graph` object that is
    equivalent to the game. The nodes of the `Graph` represent the states of `Game`,
    the edges of the `Graph` are defined by the set of `actions` and the `delta` function.
    The atomic propositions, labeling function are stored as `node, edge` and `graph` properties.
    By default, every `Graph` returned a `Game.graphify()` function have the following (node/edge/graph) properties:

    - `state`: (node property) A Map from every node to the state of transition system it represents.
    - `input_domain`: (graph property) Name of function that defines input domain of game (="actions").
    - `actions`: (graph property) List of valid actions.
    - `input`: (edge property) A map from every edge `(uid, vid, key)` to its associated action label.
    - `prob`: (edge property) The probability associated with the edge `(uid, vid, key)`.
    - `atoms`: (graph property) List of valid atomic propositions.
    - `label`: (node property) A map every node to the list of atomic propositions true in the state represented by that node.
    - `init_state`: (graph property) Initial state of transition system.
    - `is_deterministic`: (graph property) Is the transition system deterministic?
    - `is_probabilistic`: (graph property) Is the transition system probabilistic?
    - `is_turn_based`: (graph property) Is the transition system turn-based?
    - `final`: (node property) Returns an integer denoting the acceptance set the state belongs to.
    - `win_cond`: (graph property) The winning condition of the game.
    - `formula`: (graph property) A logic formula representing the winning condition of the game.
    - `turn`: (node property) A map from every node to an integer (0/1/2) that denotes which player controls the node.
    - `p1_acts`: (graph property) A subset of actions accessible to P1.
    - `p2_acts`: (graph property) A subset of actions accessible to P2.

    **Note:** Some features of probabilistic transition system are not tested.
    If you are trying to implement a probabilistic transition system, reach out to Abhishek Kulkarni
    (a.kulkarni2@ufl.edu).
    """
    NODE_PROPERTY = TSys.NODE_PROPERTY.copy()
    EDGE_PROPERTY = TSys.EDGE_PROPERTY.copy()
    GRAPH_PROPERTY = TSys.GRAPH_PROPERTY.copy()

    def __init__(self, is_turn_based=True, is_deterministic=True, is_probabilistic=False, **kwargs):
        kwargs["is_deterministic"] = is_deterministic
        kwargs["is_probabilistic"] = is_probabilistic
        super(Game, self).__init__(**kwargs)
        self._is_turn_based = is_turn_based

        # Process keyword arguments
        if "states" in kwargs:
            def states_():
                return list(kwargs["states"])

            self.states = states_

        if "actions" in kwargs:
            def actions_():
                return list(kwargs["actions"])

            self.actions = actions_

        if "trans_dict" in kwargs:
            def delta_(state, inp):
                return kwargs["trans_dict"][state][inp]

            self.delta = delta_

        if "atoms" in kwargs:
            def atoms_():
                return list(kwargs["atoms"])
            self.atoms = atoms_

        if "label" in kwargs:
            def label_(state):
                return list(kwargs["label"][state])
            self.label = label_

        if "init_state" in kwargs:
            self.initialize(kwargs["init_state"])

        if "turn" in kwargs:
            def turn_(state):
                return kwargs["turn"][state]
            self.turn = turn_

        if "final" in kwargs:
            def final_(state):
                return 0 if state in kwargs["final"] else -1

            self.final = final_

    # ==========================================================================
    # FUNCTIONS TO BE IMPLEMENTED BY USER.
    # ==========================================================================
    @register_property(NODE_PROPERTY)
    def final(self, state):
        """
        Defines whether the given state is a final state.
        The structure of final state is based on the winning condition
        [See Automata, Logics and Infinite Games (Ch. 2)].

        :param state: (object) A valid state.
        :return: (int or a list of ints). The integer denotes the acceptance set the state belongs to.
        """
        raise NotImplementedError(f"{self.__class__.__name__}.final() is not implemented.")

    @register_property(GRAPH_PROPERTY)
    def is_turn_based(self):
        """ Is the game turn based? """
        return self._is_turn_based

    @register_property(NODE_PROPERTY)
    def turn(self, state):
        """
        Defines the player who controls the given state.

        :param state: (object) A valid state.
        :return: (int). In turn-based game, turn can be 1 for player 1 or 2 for player 2.
            In concurrent games, the turn must be 0.

        .. note:: For concurrent games, the turn function can be left unimplemented.
        """
        raise NotImplementedError

    @register_property(GRAPH_PROPERTY)
    def p1_acts(self):
        """ A subset of actions accessible to P1. """
        raise NotImplementedError

    @register_property(GRAPH_PROPERTY)
    def p2_acts(self):
        """ A subset of actions accessible to P2. """
        raise NotImplementedError

    @register_property(GRAPH_PROPERTY)
    def win_cond(self):
        """ Winning condition of the game. """
        raise NotImplementedError

    @register_property(GRAPH_PROPERTY)
    def formula(self):
        """ A logic formula representing the winning condition of the game. """
        raise NotImplementedError


class Solver:
    """
    Represents a game solver that computes the winning regions and strategies for the players
    under a fixed solution concept.

    :param graph: (Graph or SubGraph instance) Graph or subgraph representing the game on a graph.
    """
    def __init__(self, graph, **kwargs):
        # Load and validate graph
        self._graph = graph
        self._solution = SubGraph(self._graph)

        # Associate node and edge properties with solution
        self._node_winner = NodePropertyMap(self._solution, default=-1)     # Values denote which player wins from node.
        self._edge_winner = EdgePropertyMap(self._solution, default=-1)     # Values denote which player wins from edge.
        self._solution["node_winner"] = self._node_winner
        self._solution["edge_winner"] = self._edge_winner

        # Status variables
        self._is_solved = False

        # Cache variables
        self._state2node = {self._solution["state"][uid]: uid for uid in self._solution.nodes()}

    def __str__(self):
        return f"<Solver for {self._graph}>"

    def graph(self):
        """ Returns the input game graph. """
        return self._graph

    def state2node(self, state):
        """ Helper function to get the node id associated with given state. """
        return self._state2node[state]

    def is_solved(self):
        """ Returns if the game is solved or not. """
        return self._is_solved

    def solution(self):
        """
        Returns the solved game graph.
        The graph contains two special properties:

        - `node_winner` (node property): Maps each node to the id of player (1/2) who wins from that node.
        - `edge_winner` (edge property): Maps each edge to the id of player (1/2) who wins using that edge.
        """
        if not self.is_solved():
            raise ValueError(f"{self} is not solved.")
        return self._solution

    def solve(self):
        """ Abstract method."""
        raise NotImplementedError

    def winner(self, state):
        """ Returns the player who wins from the given state. """
        uid = self.state2node(state)
        return self._node_winner[uid]

    def win_acts(self, state):
        """ Retuns the list of winning actions from the given state. """
        # Get the input domain (name of function) of game
        ep_input = self._solution["input"]

        # Get node id for the state
        uid = self.state2node(state)

        # Determine which player has a winning strategy at the state
        player = self._node_winner[uid]

        # Identify all winning actions.
        win_acts = set()
        for _, vid, key in self._graph.out_edges(uid):
            if self._edge_winner[uid, vid, key] == player:
                # win_acts.add(self._graph[ep_input][uid, vid, key])
                win_acts.add(ep_input[uid, vid, key])

        # Convert to list and return
        return list(win_acts)

    def win_region(self, player):
        """ Returns the winning region for the player. """
        # return [self._solution["state"][uid] for uid in self._solution.nodes() if self._node_winner[uid] == player]
        return [self._solution["state"][uid] for uid in self._solution.nodes() if self._node_winner[uid] == player]

    def reset(self):
        """ Resets the solver. """
        self._solution = SubGraph(self._graph)
        self._node_winner = NodePropertyMap(self._solution, default=-1)  # Values denote which player wins from node.
        self._edge_winner = EdgePropertyMap(self._solution, default=-1)  # Values denote which player wins from edge.
        self._solution["node_winner"] = self._node_winner
        self._solution["edge_winner"] = self._edge_winner


class Strategy:
    """
    Defines a strategy for a player.

    Allows customization of losing behavior by specifying a function that maps a losing node to an action.
    By default, losing state is mapped to None.
    """
    def __init__(self, graph, player, losing_behavior=None, **kwargs):
        assert "node_winner" in graph.node_properties(), "graph must have node property called 'node_winner'. " \
                                                         "Ensure the graph is the solution generated by a Solver."
        assert "edge_winner" in graph.edge_properties(), "graph must have edge property called 'edge_winner'. " \
                                                         "Ensure the graph is the solution generated by a Solver."
        assert losing_behavior is None or callable(losing_behavior), \
            "losing behavior should be a function that takes a state as input and returns either None or an action."

        # Instance variables
        self._graph = graph
        self._player = player
        self._strategy = dict()
        self._losing_behavior = (lambda st: None) if losing_behavior is None else losing_behavior

        # Generate and cache strategy
        self._gen_strategy()

    def _gen_strategy(self):
        raise NotImplementedError


class DeterministicStrategy(Strategy):
    """
    Represents a deterministic strategy. That is, the strategy at a given state always selects the same winning action.

    Allows customization of losing behavior by specifying a function that maps a losing node to an action.
    By default, losing state is mapped to None.
    """
    def __call__(self, state):
        """ Returns the action selected by strategy at given state. """
        return self._strategy[state]

    def _gen_strategy(self):
        # Generate a deterministic strategy by choosing an action at every winning state.
        # States that are losing for the player return according to `losing_behavior` function.
        ep_input = self._graph["input"]
        win_edges = self._graph["edge_winner"]
        for uid in self._graph.nodes():
            state = self._graph["state"][uid]
            for _, vid, key in self._graph.out_edges(uid):
                if win_edges[uid, vid, key] == self._player:
                    self._strategy[state] = self._graph[ep_input][uid, vid, key]
                    break
                self._strategy[state] = self._losing_behavior(state)


class NonDeterministicStrategy(Strategy):
    """
    Represents a non-deterministic strategy. Uniformly samples an action from winning actions at that state.

    Allows customization of losing behavior by specifying a function that maps a losing node to an action.
    By default, losing state is mapped to None.
    """

    def __call__(self, state):
        """ Returns the action selected by strategy at given state. """

        actions = self._strategy[state]
        try:
            return random.choice(actions)
        except IndexError:   # In case the list is empty.
            return None

    def _gen_strategy(self):
        # Generate a deterministic strategy by choosing an action at every winning state.
        # States that are losing for the player return according to `losing_behavior` function.
        ep_input = self._graph["input"]
        win_edges = self._graph["edge_winner"]

        # Visit all states
        for uid in self._graph.nodes():
            state = self._graph["state"][uid]

            # Initialize strategy as empty set
            self._strategy[state] = set()

            # Visit all outgoing edges from that state
            for _, vid, key in self._graph.out_edges(uid):
                # If they are winning for the player, include the action
                if win_edges[uid, vid, key] == self._player:
                    self._strategy[state].add(self._graph[ep_input][uid, vid, key])

            # If no actions were winning, follow losing behavior
            if len(self._strategy[state]) == 0:
                # User must ensure that losing behavior returns an iterable
                self._strategy[state] = self._losing_behavior(state)

            # Store the valid actions as a list
            self._strategy[state] = list(self._strategy[state])


# Currently (v0.1.5) I do not include randomized strategies with customizable probability distribution on each state.
# class RandomizedStrategy(NonDeterministicStrategy):
#     def __init__(self, graph, player, losing_behavior=None, **kwargs):
#         raise NotImplementedError("Need to figure out how to select/customize distributions at each state.")
#
#     def __call__(self, state):
#         pass
