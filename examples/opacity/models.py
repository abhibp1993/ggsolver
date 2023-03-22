"""
Models implementing paper on Opacity, CDC'23.
"""

import ggsolver.dtptb as dtptb


class ReachabilityGame(dtptb.DTPTBGame):
    """
    Defines input game. See Def. 1 + Def. 6 in paper.

    Def. 1 <=> dtptb.DTPTBGame
    attacker_observation <=> Def. 6.
    """
    EDGE_PROPERTY = dtptb.DTPTBGame.EDGE_PROPERTY.copy()

    def attacker_observation(self, state, act):
        raise NotImplementedError("Marked Abstract")


class BeliefGame(dtptb.DTPTBGame):
    def __init__(self, game: ReachabilityGame):
        super(BeliefGame, self).__init__()
        self._game = game

    def states(self):
        pass

    def turn(self, state):
        pass

    def actions(self):
        pass

    def delta(self, state, act):
        pass

    def final(self, state):
        pass
