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
    """ Def. 5 in Paper """
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
        return self._game.states()

    def actions(self):
        return self._game.actions()

    def delta(self, state, act):
        # Make all traps and fake targets a sink state
        if state in self._traps or state in self._fakes:
            return state
        else:
            return self._game.delta(state, act)

    def final(self, state):
        return self._game.final(state)

    def turn(self, state):
        return self._game.turn(state)

    def game(self):
        return self._game

    @register_property(GRAPH_PROPERTY)
    def traps(self):
        return self._traps

    @register_property(GRAPH_PROPERTY)
    def fakes(self):
        return self._fakes


class PerceptualGameP2(DecoyAllocGame):
    """ Def 6. In Paper """
    def delta(self, state, act):
        # Same as DecoyAllocGame except fakes and traps are not sink states
        return self._game.delta(state, act)

    def final(self, state):
        # TODO Unsure if _game.fakes is a set already
        return list(set(self._game.final()).union(set(self._game.fakes)))


class ReachabilityGameOfP1(Game):
    """ Def. 11 in Paper """
    def __init__(self, p2_game, solution_p2_game, traps, **kwargs):
        super(ReachabilityGameOfP1, self).__init__(is_turn_based=p2_game.is_turn_based(),
                                                   is_deterministic=p2_game.is_deterministic(),
                                                   is_probabilistic=p2_game.is_probabilistic())
        self._traps = traps
        self._p2_game = p2_game
        self._solution_p2_game = solution_p2_game

    def states(self):
        return self._solution_p2_game.win_region(2)

    def actions(self):
        return self._p2_game.actions()

    def delta(self, state, act):
        # Delta is the same if rationalizable actions and undefined otherwise
        # Def. 3/10 or Def. 5 in other paper for definition of rationalizable action
        # TODO should the action also reduce rank to be rationalizable?

        is_rationalizable = (state in self._solution_p2_game.win_region(1) and
                             self._p2_game.delta(state, act) in self._solution_p2_game.win_region(1)) \
                            or state in self._solution_p2_game.win_region(2)
        if state in self._traps:
            # Sink state for all traps
            return state
        elif is_rationalizable:
            return self._p2_game.delta(state, act)
        else:
            return None


    def final(self, state):
        return self._traps

    def turn(self, state):
        self._p2_game.turn(state)


class Hypergame(Game):
    """ Def. 18 """
    def __init__(self, p2_game, solution_p2_game, traps, fakes, **kwargs):
        super(Hypergame, self).__init__(is_turn_based=p2_game.is_turn_based(),
                                        is_deterministic=p2_game.is_deterministic(),
                                        is_probabilistic=p2_game.is_probabilistic())
        self._traps = traps
        self._fakes = fakes
        self._p2_game = p2_game
        self._solution_p2_game = solution_p2_game

    def states(self):
        return self._solution_p2_game.win_region(2)

    def actions(self):
        return self._p2_game.actions()

    def delta(self, state, act):
        is_rationalizable = (state in self._solution_p2_game.win_region(1) and
                             self._p2_game.delta(state, act) in self._solution_p2_game.win_region(1)) \
                            or state in self._solution_p2_game.win_region(2)

        if state in self._traps or state in self._fakes:
            # Sinks states for traps and fake targets
            return state
        # Check if action is rationalizable
        elif is_rationalizable:
            return self._p2_game.delta(state, act)
        else:
            return None

    def final(self, state):
        return list(set(self._traps).union(set(self._fakes)))

    def turn(self, state):
        return self._p2_game.turn(state)




