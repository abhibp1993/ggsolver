import concurrent.futures
import inspect
import multiprocessing
from tqdm import tqdm
from loguru import logger
from ggsolver.core.graph import Graph, SubGraph


# ==========================================================================
# DECORATOR FUNCTIONS.
# ==========================================================================
def register_property(property_set: set):
    def register_function(func):
        if func.__name__ in property_set:
            logger.warning(f"[WARN] Duplicate property: {func.__name__}.")
        property_set.add(func.__name__)
        return func
    return register_function


# ==========================================================================
# GRAPHICAL MODELS
# ==========================================================================
class GraphicalModel:
    NODE_PROPERTY = set()
    EDGE_PROPERTY = set()
    GRAPH_PROPERTY = set()

    # ==========================================================================
    # MAGIC METHODS
    # ==========================================================================
    def __init__(self, is_deterministic=True, is_probabilistic=False, **kwargs):
        # Types of Graphical Models.
        self._is_deterministic = is_deterministic
        self._is_probabilistic = is_probabilistic

        # Pointed model
        self._init_state = kwargs["init_state"] if "init_state" in kwargs else None

        # Cache variables
        self._cache_state2node = dict()

    @register_property(GRAPH_PROPERTY)
    def is_deterministic(self):
        """
        Returns `True` if the graphical model is deterministic. Else, returns `False`.
        """
        return self._is_deterministic

    @register_property(GRAPH_PROPERTY)
    def is_nondeterministic(self):
        """ Returns `True` if the graphical model is non-deterministic. Else, returns `False`. """
        return not self._is_deterministic and not self._is_probabilistic

    @register_property(GRAPH_PROPERTY)
    def is_probabilistic(self):
        """ Returns `True` if the graphical model is probabilistic. Else, returns `False`. """
        return self._is_probabilistic

    # ==========================================================================
    # PUBLIC FUNCTIONS.
    # ==========================================================================
    def initialize(self, s0):
        self._init_state = s0

    def make_complete(self):
        # TODO. GraphicalModel.make_complete
        # Wrap states() method to add a "__sink__" state.
        # Wrap delta() method to redirect any undefined transitions into sink state.
        pass

    def from_graph(self, graph):
        # TODO. Implement from_graph method
        # Define states()
        # Define inputs() and/or enabled_inputs()
        # Define delta() 
        # Load node properties 
        # Load edge properties
        # Load graph properties
        # Manage cached variables
        pass

    # ==========================================================================
    # FUNCTIONS TO BE IMPLEMENTED BY USER.
    # ==========================================================================
    def states(self):
        raise NotImplementedError("Abstract")

    def inputs(self):
        raise NotImplementedError("Abstract")

    def init_state(self):
        """
        Returns the initial state of the graphical model.
        """
        return self._init_state

    def delta(self, state, inp):
        raise NotImplementedError("Abstract")

    def enabled_inputs(self, state):
        raise NotImplementedError("Abstract")

    # ==========================================================================
    # GRAPHIFICATION METHODS
    # ==========================================================================
    def graphify(self, **kwargs):
        """
        Constructs the underlying graph of the graphical model.

        :param pointed: (bool) If True, constructs pointed graphical model. Otherwise, constructs complete model.
        :param cores: (int) Number of cores to use. Must be a positive integer.
            If given value is larger than number of available cores, maximum cores will be used.
        :param verbosity: (int) Accepted values:
            - 0: no messages,
            - 1: shows progress bars, critical warnings and messages. [Default]
            - 2: shows configuration, progress bars, critical warnings and messages.
            - 3: shows debug level information including delta function logging.
        :param np: (Iterable[str]) Select which node properties to include in graph.
        :param ignore_np: (Iterable[str]) Select which node properties to ignore in graph.
        :param ep: (Iterable[str]) Select which edge properties to include in graph.
        :param ignore_ep: (Iterable[str]) Select which edge properties to ignore in graph.
        :param gp: (Iterable[str]) Select which graph properties to include in graph.
        :param ignore_gp: (Iterable[str]) Select which graph properties to ignore in graph.

        :return: Equivalent graph of game with given properties.
        """
        # Process arguments
        pointed, cores, verbosity, np, ep, gp = self._graphify_process_args(kwargs)

        # Configure key functions
        states = getattr(self, "states")
        inputs = getattr(self, "inputs")
        delta = getattr(self, "delta")
        init_state = getattr(self, "init_state")
        en_inputs = getattr(self, "enabled_inputs")

        # Clear cache
        self._clear_cache()

        # Instantiate graph
        graph = Graph()

        # Initialize node and edge properties
        graph.create_np("state")
        graph.create_np("enabled_inputs")
        graph.create_ep("input")
        graph.create_ep("prob")

        # Construct underlying graph
        if not pointed and cores == 1:
            self._gen_graph_up_sc(graph=graph, states=states, inputs=inputs, delta=delta,
                                  en_inputs=en_inputs, verbosity=verbosity)
        elif not pointed and cores > 1:
            self._gen_graph_up_mc(graph=graph, states=states, inputs=inputs, delta=delta,
                                  en_inputs=en_inputs, verbosity=verbosity)
        elif pointed and cores == 1:
            self._gen_graph_p_sc(graph=graph, inputs=inputs, delta=delta, init_state=init_state,
                                 en_inputs=en_inputs, verbosity=verbosity)
        elif pointed and cores > 1:
            self._gen_graph_p_mc(graph=graph, inputs=inputs, delta=delta, init_state=init_state,
                                 en_inputs=en_inputs, verbosity=verbosity)
        else:
            raise RuntimeError(f"The following configuration is not supported: "
                               f"{pointed=}, {cores=}.")

        # Construct node, edge and graph property maps
        for p_name in np:
            self._add_np(graph, p_name, verbosity)

        for p_name in ep:
            self._add_ep(graph, p_name, verbosity)

        for p_name in gp:
            self._add_gp(graph, p_name, verbosity)

        # Return graph
        return graph

    def _clear_cache(self):
        pass

    def _gen_edges(self, delta, state, inp, verbosity):
        next_states = delta(state, inp)
        edges = set()

        # There are three types of graphical models. Handle each separately.
        # If model is deterministic, next states is a single state.
        if self.is_deterministic():
            if next_states is None and verbosity > 2:
                logger.warning(f"No edge(s) added to graph for state={state}, input={inp}, next_state={next_states}.")
                return set()

            edges.add((state, next_states, inp, None))

        # If model is non-deterministic, next states is an Iterable of states.
        elif not self.is_deterministic() and not self.is_probabilistic():
            for next_state in next_states:
                if next_state is None:
                    if verbosity > 2:
                        logger.warning(
                            f"No edge(s) added to graph for state={state}, input={inp}, next_state={next_states}.")
                    continue

                edges.add((state, next_state, inp, None))

        # If model is stochastic, next states is a Distribution of states.
        elif not self.is_deterministic() and self.is_probabilistic():
            # FIXME. I have doubts that following implementation is correct.
            #  If support is empty, the code under `if` will not execute.
            for next_state in next_states.support():
                if next_state is None:
                    if verbosity > 2:
                        logger.warning(
                            f"No edge(s) added to graph for state={state}, input={inp}, next_state={next_states}.")
                    continue

                edges.add((state, next_state, inp, next_states.pmf(next_state)))

        else:
            raise TypeError("Graphical Model is neither deterministic, nor non-deterministic, nor stochastic! "
                            f"Check the values: is_deterministic: {self.is_deterministic()}, "
                            f"self.is_quantitative:{self.is_probabilistic()}.")

        return edges

    def _gen_edges_mc(self, data):
        edges = set()
        for delta, state, inp, verbosity in data:
            edges.update(self._gen_edges(delta, state, inp, verbosity))
        return edges

    def _gen_graph_up_sc(self, graph, states, inputs, delta, en_inputs, verbosity):
        # Initialize node and edge properties
        np_state = graph["state"]
        np_enabled_inputs = graph["enabled_inputs"]
        ep_input = graph["input"]
        ep_prob = graph["prob"]

        # Get states, add them to graph, update state property and cache
        states = states()
        self._cache_state2node = dict()
        for state in tqdm(states, desc="Unpointed, single-core graphify adding nodes to graph",
                          disable=True if verbosity == 0 else False):
            sid = graph.add_node()
            self._cache_state2node[state] = sid
            np_state[sid] = state

        # If enabled inputs function is implemented, use it.
        try:
            s0 = next(iter(self._cache_state2node.keys()))
            en_inputs(s0)
        except NotImplementedError:
            en_inputs = None
            if verbosity >= 1:
                logger.warning("`enabled_inputs` function raised NotImplementedError. "
                               "Setting enabled_inputs(state) to return inputs().")

        # Otherwise, set enabled inputs to be set of all inputs.
        # If neither enabled inputs nor inputs is defined, raise exception.
        if en_inputs is None:
            # Ensure inputs() function is well-defined.
            try:
                inputs = inputs()
                graph["inputs"] = inputs
            except NotImplementedError:
                raise NotImplementedError("Neither `enabled_inputs` nor `inputs` methods are implemented. "
                                          "Terminating graphify().")

            # Redefine enabled inputs method
            def en_inputs(state_):
                return inputs

        # Generate edges
        for state, sid in tqdm(self._cache_state2node.items(), desc="Unpointed, single-core graphify adding edges",
                               disable=True if verbosity == 0 else False):
            # Get enabled inputs at state
            inputs_at_state = en_inputs(state)
            np_enabled_inputs[sid] = inputs_at_state

            # Apply each input to the state to generate next states
            for inp in inputs_at_state:
                # Generate edges from state
                new_edges = self._gen_edges(delta, state, inp, verbosity=verbosity)

                # Update graph edges
                for _, t, _, prob in new_edges:
                    tid = self._cache_state2node[t]
                    key = graph.add_edge(sid, tid)
                    ep_input[sid, tid, key] = inp
                    ep_prob[sid, tid, key] = prob

        # Log completion of this procedure
        if verbosity > 0:
            logger.success("Unpointed, single-core graphify generated underlying graph successfully.")

    def _gen_graph_up_mc(self, graph, states, inputs, delta, en_inputs, verbosity):
        # Initialize node and edge properties
        np_state = graph["state"]
        np_enabled_inputs = graph["enabled_inputs"]
        ep_input = graph["input"]
        ep_prob = graph["prob"]

        # Get states, add them to graph, update state property and cache
        states = states()
        self._cache_state2node = dict()
        for state in tqdm(states, desc="Unpointed, single-core graphify adding nodes to graph",
                          disable=True if verbosity == 0 else False):
            sid = graph.add_node()
            self._cache_state2node[state] = sid
            np_state[sid] = state

        # If enabled inputs function is implemented, use it.
        try:
            s0 = next(iter(self._cache_state2node.keys()))
            en_inputs(s0)
        except NotImplementedError:
            en_inputs = None
            if verbosity >= 1:
                logger.warning("`enabled_inputs` function raised NotImplementedError. "
                               "Setting enabled_inputs(state) to return inputs().")

        # Otherwise, set enabled inputs to be set of all inputs.
        # If neither enabled inputs nor inputs is defined, raise exception.
        if en_inputs is None:
            # Ensure inputs() function is well-defined.
            try:
                inputs = inputs()
                graph["inputs"] = inputs
            except NotImplementedError:
                raise NotImplementedError("Neither `enabled_inputs` nor `inputs` methods are implemented. "
                                          "Terminating graphify().")

            # Redefine enabled inputs method
            def en_inputs(state_):
                return inputs

        # Generate edges for each node-input pair.
        #   This is parallelized using concurrent.futures module. Each process gets a chunk of node-input pairs.
        def split_list(lst, n):
            k, m = divmod(len(lst), n)
            return [lst[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n)]

        with concurrent.futures.ProcessPoolExecutor() as executor:
            num_cpu = multiprocessing.cpu_count()
            state_input_pairs = [(delta, st, inp, verbosity) for st in self._cache_state2node.keys()
                                 for inp in en_inputs(st)]
            results = executor.map(self._gen_edges_mc, split_list(state_input_pairs, num_cpu))

        for edges in tqdm(results, desc="Unpointed graphify constructing edges."):
            for src, dst, inp, prob in edges:
                uid = self._cache_state2node[src]
                vid = self._cache_state2node[dst]
                key = graph.add_edge(uid, vid)
                ep_input[uid, vid, key] = inp
                ep_prob[uid, vid, key] = prob

    def _gen_graph_p_sc(self, graph, inputs, delta, init_state, en_inputs, verbosity):
        # Initialize node and edge properties
        np_state = graph["state"]
        np_enabled_inputs = graph["enabled_inputs"]
        ep_input = graph["input"]
        ep_prob = graph["prob"]

        # Initialize state cache
        self._cache_state2node = dict()

        # Add initial state
        try:
            s0 = init_state()
            uid = graph.add_node()
            self._cache_state2node[s0] = uid
            np_state[uid] = s0

        except NotImplementedError:
            raise NotImplementedError("`init_state` function raised NotImplementedError. "
                                      "Terminating graphify().")

        # If enabled inputs function is implemented, use it.
        try:
            en_inputs(s0)
        except NotImplementedError:
            en_inputs = None
            if verbosity >= 1:
                logger.warning("`enabled_inputs` function raised NotImplementedError. "
                               "Setting enabled_inputs(state) to return inputs().")

        # Otherwise, set enabled inputs to be set of all inputs.
        # If neither enabled inputs nor inputs is defined, raise exception.
        if en_inputs is None:
            # Ensure inputs() function is well-defined.
            try:
                inputs = inputs()
                graph["inputs"] = inputs
            except NotImplementedError:
                raise NotImplementedError("Neither `enabled_inputs` nor `inputs` methods are implemented. "
                                          "Terminating graphify().")

            # Redefine enabled inputs method
            def en_inputs(state_):
                return inputs

        # Generate edges using BFS traversal until all reachable states are visited.
        queue = [s0]
        visited = set()
        with tqdm(total=1, desc="Pointed graphify adding edges", disable=True if verbosity == 0 else False) as pbar:
            while len(queue) > 0:
                # Update progress_bar
                pbar.total = len(queue) + len(visited)
                pbar.update(1)

                # Visit a state. Add to graph. Update cache. Update node property `state`.
                state = queue.pop()
                visited.add(state)
                uid = self._cache_state2node[state]

                # Get enabled inputs at the state
                inputs_at_state = en_inputs(state)
                np_enabled_inputs[uid] = inputs_at_state

                # Apply all inputs to state
                for inp in inputs_at_state:
                    # Get successors: set of (from_st, to_st, inp, prob)
                    new_edges = self._gen_edges(delta, state, inp, verbosity)

                    for _, to_state, _, prob in new_edges:
                        # If to_state was added to queue in the past, its id will be cached.
                        # Otherwise, add new node, cache it and queue it for exploration.
                        vid = self._cache_state2node.get(to_state, None)
                        if vid is None:
                            vid = graph.add_node()
                            self._cache_state2node[to_state] = vid
                            np_state[vid] = to_state
                            queue.append(to_state)

                        # Add edge to graph
                        key = graph.add_edge(uid, vid)

                        # Set edge properties
                        ep_input[uid, vid, key] = inp
                        ep_prob[uid, vid, key] = prob

        # Log completion of this procedure
        if verbosity > 0:
            logger.success("Unpointed, single-core graphify generated underlying graph successfully.")

    def _gen_graph_p_mc(self, graph, inputs, delta, init_state, en_inputs, verbosity):
        # Initialize node and edge properties
        np_state = graph["state"]
        np_enabled_inputs = graph["enabled_inputs"]
        ep_input = graph["input"]
        ep_prob = graph["prob"]

        # Initialize state cache
        self._cache_state2node = dict()

        # Add initial state
        try:
            s0 = init_state()
            uid = graph.add_node()
            self._cache_state2node[s0] = uid
            np_state[uid] = s0

        except NotImplementedError:
            raise NotImplementedError("`init_state` function raised NotImplementedError. "
                                      "Terminating graphify().")

        # If enabled inputs function is implemented, use it.
        try:
            en_inputs(s0)
        except NotImplementedError:
            en_inputs = None
            if verbosity >= 1:
                logger.warning("`enabled_inputs` function raised NotImplementedError. "
                               "Setting enabled_inputs(state) to return inputs().")

        # Otherwise, set enabled inputs to be set of all inputs.
        # If neither enabled inputs nor inputs is defined, raise exception.
        if en_inputs is None:
            # Ensure inputs() function is well-defined.
            try:
                inputs = inputs()
                graph["inputs"] = inputs
            except NotImplementedError:
                raise NotImplementedError("Neither `enabled_inputs` nor `inputs` methods are implemented. "
                                          "Terminating graphify().")

            # Redefine enabled inputs method
            def en_inputs(state_):
                return inputs

        # Generate edges using BFS traversal until all reachable states are visited.
        # TODO. Change this to multiprocessing
        queue = [s0]
        visited = set()
        with tqdm(total=1, desc="Pointed graphify adding edges", disable=True if verbosity == 0 else False) as pbar:
            while len(queue) > 0:
                # Update progress_bar
                pbar.total = len(queue) + len(visited)
                pbar.update(1)

                # Visit a state. Add to graph. Update cache. Update node property `state`.
                state = queue.pop()
                visited.add(state)
                uid = self._cache_state2node[state]

                # Get enabled inputs at the state
                inputs_at_state = en_inputs(state)
                np_enabled_inputs[uid] = inputs_at_state

                # Apply all inputs to state
                for inp in inputs_at_state:
                    # Get successors: set of (from_st, to_st, inp, prob)
                    new_edges = self._gen_edges(delta, state, inp, verbosity)

                    for _, to_state, _, prob in new_edges:
                        # If to_state was added to queue in the past, its id will be cached.
                        # Otherwise, add new node, cache it and queue it for exploration.
                        vid = self._cache_state2node.get(to_state, None)
                        if vid is None:
                            vid = graph.add_node()
                            self._cache_state2node[to_state] = vid
                            np_state[vid] = to_state
                            queue.append(to_state)

                        # Add edge to graph
                        key = graph.add_edge(uid, vid)

                        # Set edge properties
                        ep_input[uid, vid, key] = inp
                        ep_prob[uid, vid, key] = prob

        # Log completion of this procedure
        if verbosity > 0:
            logger.success("Unpointed, single-core graphify generated underlying graph successfully.")

    def _add_np(self, graph, p_name, verbosity, default=None):
        try:
            p_map = graph.create_np(pname=p_name, default=default)
            p_func = getattr(self, p_name)
            if not (inspect.isfunction(p_func) or inspect.ismethod(p_func)):
                raise TypeError(f"Node property {p_func} is not a function.")
            for uid in graph.nodes():
                p_map[uid] = p_func(graph["state"][uid])

            if verbosity > 1:
                logger.success(f"Processed node property: {p_name}. [OK]")
        except NotImplementedError:
            if verbosity > 1:
                logger.warning(f"Node property function not implemented: {p_name}. [IGNORED]")
        except AttributeError:
            if verbosity > 1:
                logger.warning(f"Node property function is not defined: {p_name}. [IGNORED]")

    def _add_ep(self, graph, p_name, verbosity, default=None):
        try:
            p_map = graph.create_ep(pname=p_name, default=default)
            p_func = getattr(self, p_name)
            if not (inspect.isfunction(p_func) or inspect.ismethod(p_func)):
                raise TypeError(f"Edge property {p_func} is not a function.")
            for uid, vid, key in graph.edges():
                p_map[uid, vid, key] = p_func(graph["state"][uid], graph["input"][uid, vid, key], graph["state"][vid])

            if verbosity > 1:
                logger.success(f"Processed node property: {p_name}. [OK]")
        except NotImplementedError:
            if verbosity > 1:
                logger.warning(f"Node property function not implemented: {p_name}. [IGNORED]")
        except AttributeError:
            if verbosity > 1:
                logger.warning(f"Node property function is not defined: {p_name}. [IGNORED]")

    def _add_gp(self, graph, p_name, verbosity):
        try:
            p_func = getattr(self, p_name)
            if inspect.ismethod(p_func) or (inspect.isfunction(p_func) and p_func.__name__ == "<lambda>"):
                graph[p_name] = p_func()
                if verbosity > 1:
                    logger.info(f"Processed graph property: {p_name}. [OK]")

            elif inspect.isfunction(p_func):
                if len(inspect.signature(p_func).parameters) == 0:
                    graph[p_name] = p_func()
                else:
                    graph[p_name] = p_func(self)

                if verbosity > 1:
                    logger.info(f"Processed graph property: {p_name}. [OK]")
            else:
                raise TypeError(f"Graph property {p_name} is neither a function nor a method.")
        except NotImplementedError:
            if verbosity > 1:
                logger.warning(f"Graph property is not implemented: {p_name}. [IGNORED]")
        except AttributeError:
            if verbosity > 1:
                logger.warning(f"Node property function is not defined: {p_name}. [IGNORED]")

    def _graphify_process_args(self, kwargs):
        pointed = kwargs.get("pointed", False)
        assert isinstance(pointed, bool)

        cpu_cores = multiprocessing.cpu_count()
        cores = max(1, min(cpu_cores, kwargs.get("cores", 1)))

        verbosity = kwargs.get("verbosity", 1)
        if verbosity < 0 or verbosity > 3:
            verbosity = 1

        np = set(kwargs.get("np", set()))
        if np is None:
            np = getattr(self, "NODE_PROPERTY")
        else:
            np = set(getattr(self, "NODE_PROPERTY")).intersection(set(np))

        ignore_np = set(kwargs.get("ignore_np", set()))
        np = np - ignore_np

        ep = set(kwargs.get("ep", set()))
        if ep is None:
            ep = getattr(self, "EDGE_PROPERTY")
        else:
            ep = set(getattr(self, "EDGE_PROPERTY")).intersection(set(ep))

        ignore_ep = set(kwargs.get("ignore_ep", set()))
        ep = ep - ignore_ep

        gp = set(kwargs.get("gp", set()))
        if gp is None:
            gp = getattr(self, "GRAPH_PROPERTY")
        else:
            gp = set(getattr(self, "GRAPH_PROPERTY")).intersection(set(gp))

        ignore_gp = set(kwargs.get("ignore_gp", set()))
        gp = gp - ignore_gp

        # Log information
        if verbosity >= 2:
            logger.info(f"Graphify configuration: {pointed=}.")
            logger.info(f"Graphify configuration: {cores=}.")
            logger.info(f"Graphify configuration: {verbosity=}.")
            logger.info(f"Graphify configuration: {np=}.")
            logger.info(f"Graphify configuration: {ep=}.")
            logger.info(f"Graphify configuration: {gp=}.")

            np_ignore = set(getattr(self, "NODE_PROPERTY")).symmetric_difference(set(np))
            ep_ignore = set(getattr(self, "EDGE_PROPERTY")).symmetric_difference(set(ep))
            gp_ignore = set(getattr(self, "GRAPH_PROPERTY")).symmetric_difference(set(gp))
            logger.info(f"Graphify configuration: Ignored node properties: {np_ignore}.")
            logger.info(f"Graphify configuration: Ignored edge properties: {ep_ignore}.")
            logger.info(f"Graphify configuration: Ignored graph properties: {gp_ignore}.")

        # Always log to debug
        logger.debug(f"Graphify configuration: {pointed=}.")
        logger.debug(f"Graphify configuration: {cores=}.")
        logger.debug(f"Graphify configuration: {verbosity=}.")
        logger.debug(f"Graphify configuration: {np=}.")
        logger.debug(f"Graphify configuration: {ep=}.")
        logger.debug(f"Graphify configuration: {gp=}.")

        np_ignore = set(getattr(self, "NODE_PROPERTY")).symmetric_difference(set(np))
        ep_ignore = set(getattr(self, "EDGE_PROPERTY")).symmetric_difference(set(ep))
        gp_ignore = set(getattr(self, "GRAPH_PROPERTY")).symmetric_difference(set(gp))
        logger.debug(f"Graphify configuration: Ignored node properties: {np_ignore}.")
        logger.debug(f"Graphify configuration: Ignored edge properties: {ep_ignore}.")
        logger.debug(f"Graphify configuration: Ignored graph properties: {gp_ignore}.")

        return pointed, cores, verbosity, np, ep, gp

    def _graphify_preprocess_properties(self, np, ep, gp, verbosity):
        pass


class Game(GraphicalModel):
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
    **Note:** Some features of probabilistic transition system are not tested.
    If you are trying to implement a probabilistic transition system, reach out to Abhishek Kulkarni
    (a.kulkarni2@ufl.edu).
    """
    NODE_PROPERTY = GraphicalModel.NODE_PROPERTY.copy()
    EDGE_PROPERTY = GraphicalModel.EDGE_PROPERTY.copy()
    GRAPH_PROPERTY = GraphicalModel.GRAPH_PROPERTY.copy()

    def __init__(self, is_turn_based=True, is_deterministic=True, is_probabilistic=False, **kwargs):
        kwargs["is_deterministic"] = is_deterministic
        kwargs["is_probabilistic"] = is_probabilistic
        super(Game, self).__init__(**kwargs)
        self._is_turn_based = is_turn_based

        # Aliases of special methods
        self.inputs = self.actions
        self.enabled_inputs = self.enabled_acts

        # Process keyword arguments
        self.initialize(kwargs.get("init_state", None))

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
    @register_property(GRAPH_PROPERTY)
    def actions(self):
        """
        Defines the actions component of the transition system.
        :return: (list/tuple of str). List or tuple of action labels.
        """
        raise NotImplementedError(f"{self.__class__.__name__}.actions() is not implemented.")

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
        #   Values denote which player wins from node, edge.
        self._node_winner = self._solution.create_np(pname="node_winner", default=-1)
        self._edge_winner = self._solution.create_ep(pname="edge_winner", default=-1)

        # Status variables
        self._is_solved = False

        # Cache variables
        self._cache_state2node = {self._solution["state"][uid]: uid for uid in self._solution.nodes()}

    def __str__(self):
        return f"<Solver for {self._graph}>"

    def graph(self):
        """ Returns the input game graph. """
        return self._graph

    def state2node(self, state):
        """ Helper function to get the node id associated with given state. """
        return self._cache_state2node[state]

    def node2state(self, uid):
        """ Helper function to get the state associated with node id. """
        return self._graph["state"][uid]

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

    def state_winner(self, state):
        """ Returns the player who wins from the given state. """
        uid = self.state2node(state)
        return self._node_winner[uid]

    def node_winner(self, uid):
        """ Returns the player who wins from the given state. """
        return self._node_winner[uid]

    def winning_actions(self, state):
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

    def winning_states(self, player):
        """ Returns the winning region for the player. """
        # return [self._solution["state"][uid] for uid in self._solution.nodes() if self._node_winner[uid] == player]
        return [self._solution["state"][uid] for uid in self._solution.nodes() if self._node_winner[uid] == player]

    def winning_edges(self, uid):
        """ Returns the list of winning actions from the given node. """
        # Determine which player has a winning strategy at the node
        player = self._node_winner[uid]

        # Identify all winning actions.
        win_edges = set()
        for _, vid, key in self._graph.out_edges(uid):
            if self._edge_winner[uid, vid, key] == player:
                win_edges.add((uid, vid, key))

        # Convert to list and return
        return list(win_edges)

    def winning_nodes(self, player):
        """ Returns the winning region for the player. """
        return [uid for uid in self._solution.nodes() if self._node_winner[uid] == player]

    def reset(self):
        """ Resets the solver. """
        self._solution = SubGraph(self._graph)
        self._node_winner = self._solution.create_np(pname="node_winner", default=-1, overwrite=True)
        self._edge_winner = self._solution.create_ep(pname="edge_winner", default=-1, overwrite=True)


def cached(model: GraphicalModel):
    # TODO. Implement cached
    # Add _cache_delta variable to model.
    # Wrap model.delta in custom function that manages the cache variable.
    # Update model.delta
    # Return
    pass


if __name__ == '__main__':
    m = Game()
    m.graphify()
