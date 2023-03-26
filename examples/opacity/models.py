"""
Models implementing paper on Opacity, CDC'23.
"""
import itertools

import ggsolver.util as util
import ggsolver.dtptb as dtptb
import ggsolver.logic as logic
import ggsolver.models as models


class Arena(dtptb.DTPTBGame):
    """
    Defines input game. See Def. 1 + Def. 6 in paper.
    Def. 1 <=> dtptb.DTPTBGame
    attacker_observation <=> Def. 6.
    """
    EDGE_PROPERTY = dtptb.DTPTBGame.EDGE_PROPERTY.copy()

    @models.register_property(EDGE_PROPERTY)
    def attacker_observation(self, state, act, next_state):
        raise NotImplementedError("Marked Abstract")


class BeliefGame(dtptb.DTPTBGame):
    def __init__(self, game: Arena, aut: logic.automata.DFA):
        super(BeliefGame, self).__init__()
        self._game = game
        self._aut = aut

    # def states(self):
    #     T = itertools.product(self._game.states(), self._aut.states())
    #     print(f"constructing powerset, {len(list(T))}")
    #     PT = util.powerset(T)
    #     # print(len(PT))
    #     print(len(list(itertools.product(T,PT))))
    #     return list(itertools.product(T,PT))

    def turn(self, state):
        return self._game.turn(state[0])

    def actions(self):
        return self._game.actions()

    def delta(self, state, act):
        s, q, b = state
        t = self._game.delta(s, act)
        p = self._aut.delta(q, self._game.label(t))

        if t is None:
            return

        c = set()
        o = self._game.attacker_observation(s, act, t)
        for (s_b, q_b), a_b in itertools.product(b, self.actions()):
            t_b = self._game.delta(s_b, a_b)
            p_b = self._aut.delta(q_b, self._game.label(t_b))
            if o == self._game.attacker_observation(s_b, a_b, t_b):
                c.add((t_b, p_b))
        return t, p, tuple(c)

    def final(self, state):
        s, q, b = state
        if_final = 0 in self._aut.final(q)
        if any(0 not in self._aut.final(q_b) for s_b, q_b in b):
            return if_final
        return False
        # for s_b, q_b in b:
        #     if_final = if_final and (0 in self._aut.final(q_b))
        # return if_final

    def init_state(self):
        s0 = self._game.init_state()
        q0 = self._aut.init_state()
        return s0, q0, ((s0, q0),)
