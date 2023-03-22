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
    def obs_set(self):
        """
        Implement algorithm to apply query from each state.
        :return:
        """
        raise NotImplementedError("TODO. ")

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
