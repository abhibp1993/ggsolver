import enum

import networkx as nx
from prefltlf2pdfa import PrefAutomaton

from ggsolver import TSys, GameGraph, NewStateCls

ProdState = NewStateCls("ProdState",
                        ["game_state", "aut_state"])


class StochasticOrderType(enum.Enum):
    WEAK = "weak"
    WEAK_STAR = "weak*"
    STRONG = "strong"


class ProductGame(TSys):
    def __init__(self, game: TSys | GameGraph, aut: PrefAutomaton):
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


class PrefGameGraph(GameGraph):
    def __init__(self, graph: nx.MultiDiGraph, aut: PrefAutomaton):
        super().__init__(graph)
        self._aut = aut

    @property
    def aut(self):
        return self._aut
