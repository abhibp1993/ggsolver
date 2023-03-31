"""
Models implementing paper on Opacity, CDC'23.
"""
import itertools
import logging
# import loguru

import ggsolver.util as util
import ggsolver.dtptb as dtptb
import ggsolver.logic as logic
import ggsolver.models as models

logging.basicConfig(filename="out/belief.log", level=logging.DEBUG)


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
        self._cache = dict()        # maps {state, {act: n_state}}

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

    def enabled_acts(self, state):
        s, q, b = state
        return self._game.enabled_acts(s)

    def delta(self, state, act):
        s, q, b = state
        if 0 in self._aut.final(q):
            return state

        # Check cache. If unavailable, use game.delta
        if s in self._cache and act in self._cache[s]:
            t = self._cache[s][act]
        else:
            t = self._game.delta(s, act)
            if s in self._cache:
                self._cache[s][act] = t
            else:
                self._cache[s] = {act: t}

        p = self._aut.delta(q, self._game.label(t))

        if t is None:
            return

        c = set()
        o = self._game.attacker_observation(s, act, t)
        for (s_b, q_b), a_b in itertools.product(b, self.actions()):
            if a_b not in self._game.enabled_acts(s_b):
                continue

            if s_b in self._cache and a_b in self._cache[s_b]:
                t_b = self._cache[s_b][a_b]
            else:
                t_b = self._game.delta(s_b, a_b)
                if s_b in self._cache:
                    self._cache[s_b][a_b] = t_b
                else:
                    self._cache[s_b] = {a_b: t_b}

            p_b = self._aut.delta(q_b, self._game.label(t_b))
            if o == self._game.attacker_observation(s_b, a_b, t_b):
                c.add((t_b, p_b))

        # PATCH
        if len(c) > 10:
            with open("belief.log", "a") as file:
                file.writelines([f"\n{state=} \n{act=}\n"] + [f"\t{st}\n" for st in c])
            # logging.warning(util.ColoredMsg.warn(
            #     f"\nGame({self._game.init_state()}) {state=} {act=} belief:{c}")
            # )

        return t, p, tuple(sorted(list(c)))

    def final(self, state):
        s, q, b = state
        if_final = 0 in self._aut.final(q)
        if any(0 not in self._aut.final(q_b) for s_b, q_b in b):
            return if_final
        return False

    def final_p2(self, state):
        s, q, b = state
        if_final = 0 in self._aut.final(q)
        if all(0 in self._aut.final(q_b) for s_b, q_b in b):
            return if_final
        return False

    def init_state(self):
        s0 = self._game.init_state()
        q0 = self._aut.init_state()
        return s0, q0, ((s0, q0),)
