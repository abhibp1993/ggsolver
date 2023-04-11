import ggsolver.dtptb as dtptb


class L2HypergameTraps(dtptb.DTPTBGame):
    def __init__(self, game, solution, **kwargs):
        super(L2HypergameTraps, self).__init__()
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