"""
Implements the paper on Decoy Allocation Games on Graphs.
"""

from ggsolver.models import Game, register_property


class ReachabilityGame(Game):
    """
    Def. 1 in paper.
    """
    def __init__(self, **kwargs):
        """
        Supported keyword arguments:
        - "states": a list of states.
        - "actions": a list of actions.
        - "trans": a dictionary of {s: {a: t}}, where `s, t` are states and `a` is an action.
        - "final": a list of final states.
        - "turn": a function that inputs a node and returns either 1 or 2 to indicate whether
          node is controlled by P1 or P2 node.
        """
        super(ReachabilityGame, self).__init__(is_turn_based=True, is_deterministic=True, is_probabilistic=False)

        if "states" in kwargs:
            self.states = lambda: list(kwargs["states"])

        if "actions" in kwargs:
            self.actions = lambda: list(kwargs["actions"])

        if "final" in kwargs:
            self.final = lambda n: n in kwargs["final"]

        if "turn" in kwargs:
            self.turn = kwargs["turn"]

        if "trans" in kwargs:
            def tmp_delta(s, a):
                try:
                    return kwargs["trans"][s][a]
                except KeyError:
                    return None
            self.delta = tmp_delta


class DecoyAllocGame(Game):
    GRAPH_PROPERTY = Game.GRAPH_PROPERTY.copy()
    NODE_PROPERTY = Game.NODE_PROPERTY.copy()

    def __init__(self, game, traps, fakes, **kwargs):
        super(DecoyAllocGame, self).__init__(is_turn_based=game.is_turn_based(),
                                             is_deterministic=game.is_deterministic(),
                                             is_probabilistic=game.is_probabilistic())

        self._game = game
        self._traps = traps
        self._fakes = fakes

    def states(self):
        pass

    def actions(self):
        pass

    def delta(self, state, act):
        pass

    def final(self, state):
        pass

    def turn(self, state):
        pass

    def game(self):
        return self._game

    @register_property(GRAPH_PROPERTY)
    def traps(self):
        return self._traps

    @register_property(GRAPH_PROPERTY)
    def fakes(self):
        return self._fakes


class PerceptualGameP2(DecoyAllocGame):
    def delta(self, state, act):
        pass

    def final(self, state):
        pass


class ReachabilityGameOfP1(Game):
    def __init__(self, p2_game, solution_p2_game, **kwargs):
        super(ReachabilityGameOfP1, self).__init__(is_turn_based=p2_game.is_turn_based(),
                                                   is_deterministic=p2_game.is_deterministic(),
                                                   is_probabilistic=p2_game.is_probabilistic())

        self._p2_game = p2_game
        self._solution_p2_game = solution_p2_game

    def states(self):
        pass

    def actions(self):
        pass

    def delta(self, state, act):
        pass

    def final(self, state):
        return self._traps

    def turn(self, state):
        pass


class Hypergame(Game):
    def __init__(self, p2_game, solution_p2_game, **kwargs):
        super(Hypergame, self).__init__(is_turn_based=p2_game.is_turn_based(),
                                        is_deterministic=p2_game.is_deterministic(),
                                        is_probabilistic=p2_game.is_probabilistic())

        self._p2_game = p2_game
        self._solution_p2_game = solution_p2_game

    def states(self):
        pass

    def actions(self):
        pass

    def delta(self, state, act):
        pass

    def final(self, state):
        pass

    def turn(self, state):
        pass




