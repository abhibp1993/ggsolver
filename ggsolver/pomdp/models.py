import ggsolver.models as models
import ggsolver.logic.automata as automata
import ggsolver.mdp as mdp
import itertools

from ggsolver.util import powerset


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


"""
Author: Sumukha Udupa.
Paper: CITE Opacity-enforcing active perception and control against eavesdropping attacks.

"""


class ActivePOMDP(models.Game):
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

    # TODO. Merge into obs_set: {(s, sigma): [P1 obs, P2 obs]}
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
        s, q = state
        next_states = list()
        t = self._game.delta(s, act)
        for st in t:
            p = self._aut.delta(q, self._game.label(st))
            next_states.append((st, p))
        return next_states

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
        return itertools.product(self._game.init_observation(), self._aut.states()), itertools.product(
            self._game.states(), self._aut.states())


class OpacityEnforcingGame(mdp):

    def __init__(self, game: ProductWithDFA):
        super(OpacityEnforcingGame, self).__init__()
        self._game = game

    # def states(self):
    #     states = list()
    #     belief_list = list()
    #
    #     for (state, query), value in self._game.obs_set_1():
    #
    #         power_set_of_belief = powerset(value)
    #         for B in power_set_of_belief:
    #             if state in B:
    #                 belief_list.append(B)
    #
    #         for B1, B2 in itertools.product(belief_list, belief_list):
    #             states.append(
    #                 (state, B1, B2))  # TODO: Check if B1 and B2 should be sent in as list itself or as sets/frozensets?
    #
    #         belief_list = list()
    #
    #     return states
    def states(self):
        raise NotImplementedError("Due to exploding belief states, only pointed graphify must be used.")

    def actions(self):
        return itertools.product(self._game.actions(), self._game.sensor_query())

    def init_state(self):
        initial_obs = self._game.init_observation()
        initial_state = (self._game.init_state(), initial_obs[0], initial_obs[1])
        return initial_state

    def final(self, state):  # TODO: Change if changes are made in states()
        # TODO. Rename variables PEP8 conventions.
        st, B1, B2 = state
        p1_flag = 1
        p2_flag = 1

        for s in B1:
            if self._game.final(s) == 0:
                p1_flag = 0
                break

        if p1_flag == 0:
            return False

        for state in B2:
            if self._game.final(state) == 0:
                p2_flag = 0
                break

        if p1_flag == 1 and p2_flag == 0:
            return True
        else:
            return False
    # TODO. Check return 0, 1 -> True, False

    def belief_one_dash(self, belief, action):
        a, X = action
        post_belief = list()
        for s in belief:
            post = self._game.delta(s, a)
            post_belief.append(post)

        return post_belief

    def belief_two_dash(self, belief):
        post_belief = list()
        for a in self._game.actions():
            for s in belief:
                post = self._game.delta(s, a)
                post_belief.append(post)

        return post_belief

    def delta(self, state, action):
        a, X = action
        delta_states = list()
        if self.final(state) == 0:      # TODO. final -> true, false.
            st, B1, B2 = state
            next_states = self._game.delta(st, a)
            post_b1 = self.belief_one_dash(B1, action)
            post_b2 = self.belief_two_dash(B2)

            for nx_st in next_states:
                observation_1, observation_2 = self._game.observation(nx_st, X)
                B1_dash = set(post_b1).intersection(set(observation_1))
                B2_dash = set(post_b2).intersection(set(observation_2))
                delta_states.append((nx_st, list(B1_dash), list(B2_dash)))

        else:
            delta_states.append(state)

        return delta_states
