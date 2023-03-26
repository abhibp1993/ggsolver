import random
# from ggsolver.models import Solver
import ggsolver.models as models
import itertools
from tqdm import tqdm


class ASWinReach(models.Solver):
    def __init__(self, graph, final=None, player=1, **kwargs):
        """
        Instantiates a sure winning reachability game solver.

        :param graph: (Graph instance)
        :param final: (iterable) A list/tuple/set of final nodes in graph.
        :param player: (int) Either 1 or 3.
        :param kwargs: ASureWinReach accepts no keyword arguments.
        """
        super(ASWinReach, self).__init__(graph, **kwargs)
        self._player = player
        self._final = set(final) if final is not None else {n for n in graph.nodes() if self._graph["final"][n] == 0}

    def inner_loop(self, level_y):
        """
        Inner loop of Alg.1 of the paper.
        :param
        """
        level_r = list()

        final_states = copy.deepcopy(self._final)
        level_r.append(set(final_states))
        while True:
            top_level_r = level_r[-1]
            new_level_r = set()

            p1_states = self.progress(top_level_r, level_y)

            new_level_r = set.union(top_level_r, p1_states)

            print("Executing inner_loop.")

            if top_level_r == new_level_r:
                break
            else:
                level_r.append(new_level_r)

        return new_level_r

    def solve(self):
        """
        Opacity-Enforcing Almost-Sure Winning Region
        """
        # TODO. self._solution -> SubGraph.

        level_y = list()
        level_y.append(set(self.graph.nodes()))

        while True:
            top_level_y = level_y[-1]
            new_level_y = self.inner_loop(top_level_y)

            print("Solving the one-player stochastic game.")

            if top_level_y == new_level_y:
                break
            else:
                level_y.append(new_level_y)

        strategy_1 = self.strategy_construction(new_level_y)



        # Mark the game to be solved
        self._is_solved = True

    def progress(self, set_r, set_y):
        progressive_transition_states = set()
        for q in set(self.graph.nodes()).intersection(set_y):
            allowed_action = self.allow_equivalent(q, set_y)
            if len(allowed_action) != 0:
                for m in allowed_action:
                    post_states = {vid for _, vid, m in self.graph.out_edges(q)}
                    # post_states = set(self.graph.delta(q, m))
                    if len(post_states.intersection(set_r)) != 0:
                        progressive_transition_states.add(q)
                        break

        return progressive_transition_states

    def allow(self, uid, set_y):

        # Algorithm.
        # 1. Iterate over all out_edges.
        # 2.   If vid is in set_y.
        # 3.     Add action_label:self.graph["input"][uid, vid, key] to allowed_actions
        # 4.   Else.
        # 5.     Add action_label to reject_actions
        # 6. Return allowed_actions - reject_actions

        allowed_actions = set()
        rejected_actions = set()

        for _, vid, key in self.graph.out_edges(uid):
            if vid in set_y:
                allowed_actions.add(
                    self.graph["input"][uid, vid, key])  # TODO: Check if the action input has been accessed properly.
            else:
                rejected_actions.add(self.graph["input"][uid, vid, key])

        final_allowed_actions = allowed_actions - rejected_actions
        return final_allowed_actions

        # actions = {key for _, vid, key in self.graph.out_edges(uid)}
        # for act in actions:
        #     if len(self.graph.delta(uid, act)) != 0:
        #         if set(self.graph.delta(uid, act)).issubset(set_y):
        #             allowed_actions.add(act)
        # return allowed_actions

    def observation_equivalent_states(self, uid):
        """
           Function identifies the observation equivalent states.
           :param self: Obtain the graph
           :param current_state: the current node for which we find equivalent states.
           :return: observation equivalent states list.
           """

        observation_equivalent_states = list()
        for items in (self.graph["equivalence_cls"][st] for st in self.graph.nodes()):
            if uid in items:
                observation_equivalent_states = items
                break
        return observation_equivalent_states

        # observation_equivalent_states = list()
        # st, b1, b2 = self.graph["state"][uid]
        #
        # for x in b1:
        #     if (x, b1, b2) in (self.graph["state"][uid] for uid in self.graph.nodes()):
        #         observation_equivalent_states.append((x, b1, b2))
        #     else:
        #         print("Error - Obs equivalent state not in state space.")
        #
        # return observation_equivalent_states

    def allow_equivalent(self, uid, set_y):
        allowed_actions = set()
        m = 1
        obs_eqv_states = self.observation_equivalent_states(uid)
        for uid_eqv in obs_eqv_states:
            allowed = self.allow(uid_eqv, set_y)
            if m == 1:
                allowed_actions = allowed
                m = m + 1
            else:
                allowed_actions = allowed_actions.intersection(allowed)

        return allowed_actions

    # def delta(self, state, action):
    #     return delta_states
