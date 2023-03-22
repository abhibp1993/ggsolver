import ggsolver.models as models
import ggsolver.logic.automata as automata
import itertools


class QualPOMDP(models.Game):
    """
    delta(s, a) -> [s]
    """
    GRAPH_PROPERTY = models.Game.GRAPH_PROPERTY.copy()
    NODE_PROPERTY = models.Game.NODE_PROPERTY.copy()

    def __init__(self, **kwargs):
        """
        kwargs:
            * states: List of states
            * actions: List of actions
            * trans_dict: Dictionary of {state: {act: List[state]}}
            * atoms: List of atoms
            * label: Dictionary of {state: List[atoms]}
            * final: List of states
        """
        # kwargs = filter_kwargs(states, actions, trans_dict, init_state, final)
        super(QualPOMDP, self).__init__(
            **kwargs,
            is_deterministic=False,
            is_probabilistic=False,
            is_turn_based=False
        )

    @models.register_property(GRAPH_PROPERTY)
    def obs_set(self):
        raise NotImplementedError("Marked Abstract")

    @models.register_property(NODE_PROPERTY)
    def observation(self, state):
        raise NotImplementedError("Marked Abstract")


class ActivePOMDP(models.Game):
    """
    Opacity work of Sumukha. CITE PAPER!
    delta(s, a) -> [s]
    """
    GRAPH_PROPERTY = models.Game.GRAPH_PROPERTY.copy()
    NODE_PROPERTY = models.Game.NODE_PROPERTY.copy()

    def __init__(self, **kwargs):
        """
        kwargs:
            * states: List of states
            * actions: List of actions
            * trans_dict: Dictionary of {state: {act: List[state]}}
            * atoms: List of atoms
            * label: Dictionary of {state: List[atoms]}
            * final: List of states
        """
        # kwargs = filter_kwargs(states, actions, trans_dict, init_state, final)
        super(ActivePOMDP, self).__init__(
            **kwargs,
            is_deterministic=False,
            is_probabilistic=False,
            is_turn_based=False
        )

    @models.register_property(GRAPH_PROPERTY)
    def obs_set_1(self):  # CHECK: Returns a dictionary.
        """
        Implement algorithm to apply query from each state for P1.
        :return:
        """
        observation_set_1 = dict()
        for st, query in itertools.product(self.states(), self.sensor_query()):
            observation_set_1[(st, query)] = self.observation(st, query)
        return observation_set_1

    @models.register_property(GRAPH_PROPERTY)
    def obs_set_2(self):  # CHECK: Returns a dictionary.
        """
        Implement algorithm to apply query from each state for P2.
        :return:
        """
        observation_set_2 = dict()
        for st, query in itertools.product(self.states(), self.sensor_query()):
            unsecured_sensors_queried = query.intersection(set(self.sensors_unsecured()))
            observation_set_2[(st, query)] = self.observation(st, unsecured_sensors_queried)
        return observation_set_2

    @models.register_property(GRAPH_PROPERTY)
    def sensors(self):
        raise NotImplementedError("Marked Abstract")

    @models.register_property(GRAPH_PROPERTY)
    def sensors_secured(self):
        raise NotImplementedError("Marked Abstract")

    @models.register_property(GRAPH_PROPERTY)
    def sensors_unsecured(self):
        raise NotImplementedError("Marked Abstract")

    @models.register_property(GRAPH_PROPERTY)
    def sensor_query(self):
        raise NotImplementedError("Marked Abstract")

    @models.register_property(GRAPH_PROPERTY)
    def init_observation(self):
        raise NotImplementedError("Marked Abstract")

    def observation(self, state, query):
        raise NotImplementedError("Marked Abstract")


# Check if model is ActivePOMDP.
class ProductWithDFA(ActivePOMDP):
    """
    For the product to be defined, Game must implement `atoms` and `label` functions.
    """

    def __init__(self, game: ActivePOMDP, aut: automata.DFA):
        super(ProductWithDFA, self).__init__()
        self._game = game
        self._aut = aut

    def states(self):
        return list(itertools.product(self._game.states(), self._aut.states()))

    def actions(self):
        return self._game.actions()

    def delta(self, state, act):
        # TODO. Change the definition.
        s, q = state
        t = self._game.delta(s, act)
        p = self._aut.delta(q, self._game.label(t))
        return t, p

    def init_state(self):
        if self._game.init_state() is not None:
            s0 = self.init_state()
            q0 = self._aut.init_state()
            return s0, self._aut.delta(q0, self._game.label(s0))

    def final(self, state):
        return 0 in self._aut.final(state[1])

    def observation(self, state, query):
        s, q = state
        unsecured_sensors = query.intersection(set(self._game.sensors_unsecured()))
        observation_1 = itertools.product(self._game.observation(s, query), self._aut.states())
        observation_2 = itertools.product(self._game.observation(s, unsecured_sensors), self._aut.states())

        return observation_1, observation_2

    def init_observation(self):
        return itertools.product(self._game.init_observation(), self._aut.states()), itertools.product(self._game.states(), self._aut.states())

    
