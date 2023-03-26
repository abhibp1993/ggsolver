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
        self._sensors = kwargs["sensors"] if "sensors" in kwargs else None
        self._sensors_secured = kwargs["sensors_secured"] if "sensors_secured" in kwargs else None
        self._sensors_unsecured = kwargs["sensors_unsecured"] if "sensors_unsecured" in kwargs else None
        self._sensor_query = kwargs["sensor_query"] if "sensor_query" in kwargs else None
        self._init_observation = kwargs["init_observation"] if "init_observation" in kwargs else None

    # TODO. Merge into obs_set: {(s, sigma): [P1 obs, P2 obs]}
    @models.register_property(GRAPH_PROPERTY)
    def obs_set_1(self):  # CHECK: Returns a dictionary.
        """
        Implement algorithm to apply query from each state for P1.
        :return:
        """
        observation_set = dict()
        for st, query in itertools.product(self.states(), self.sensor_query()):
            unsecured_sensors = set(query).intersection(set(self.sensors_unsecured()))
            observation_p1 = self.observation(st, query)
            observation_p2 = self.observation(st, unsecured_sensors)
            observation_set[(st, query)] = [observation_p1, observation_p2]
        return observation_set

    # @models.register_property(GRAPH_PROPERTY)
    # def obs_set_2(self):  # CHECK: Returns a dictionary.
    #     """
    #     Implement algorithm to apply query from each state for P2.
    #     :return:
    #     """
    #     observation_set_2 = dict()
    #     for st, query in itertools.product(self.states(), self.sensor_query()):
    #         unsecured_sensors_queried = query.intersection(set(self.sensors_unsecured()))
    #         observation_set_2[(st, query)] = self.observation(st, unsecured_sensors_queried)
    #     return observation_set_2

    @models.register_property(GRAPH_PROPERTY)
    def sensors(self):
        return self._sensors

    @models.register_property(GRAPH_PROPERTY)
    def sensors_secured(self):
        return self._sensors_secured

    @models.register_property(GRAPH_PROPERTY)
    def sensors_unsecured(self):
        return self._sensors_unsecured

    @models.register_property(GRAPH_PROPERTY)
    def sensor_query(self):
        return self._sensor_query

    @models.register_property(GRAPH_PROPERTY)
    def init_observation(self):
        return self._init_observation

    def observation(self, state, query):
        obs = {st for st in self.states()}

        for act in query:
            if state in self.sensors()[act]:
                obs = obs.intersection(set(self.sensors()[act]))
            else:
                obs = obs.intersection(set(self.states()) - set(self.sensors()[act]))

        return list(obs)


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
            s0 = self._game.init_state()
            q0 = self._aut.init_state()
            return s0, self._aut.delta(q0, self._game.label(s0))

    def final(self, state):
        return 0 in self._aut.final(state[1])

    # @models.register_property(GRAPH_PROPERTY)
    def sensors(self):
        return self._game.sensors()

    # @models.register_property(GRAPH_PROPERTY)
    def sensors_secured(self):
        return self._game.sensors_secured()

    # @models.register_property(GRAPH_PROPERTY)
    def sensors_unsecured(self):
        return self._game.sensors_unsecured()

    # @models.register_property(GRAPH_PROPERTY)
    def sensor_query(self):
        return self._game.sensor_query()

    def observation(self, state, query):
        s, q = state
        unsecured_sensors = set(self.sensor_query()[query]).intersection(set(self.sensors_unsecured()))
        # unsecured_sensors = query.intersection(set(self._game.sensors_unsecured()))
        observation_1 = itertools.product(self._game.observation(s, set(self.sensor_query()[query])), self._aut.states())
        observation_2 = itertools.product(self._game.observation(s, unsecured_sensors), self._aut.states())

        return list(observation_1), list(observation_2)

    def init_observation(self):
        return [list(self.init_state()), list(itertools.product(
            self._game.states(), [self.init_state()[1]]))]
        # return [list(self.init_state()), list(itertools.product(
        #     self._game.states(), self._aut.states()))]


class OpacityEnforcingGame(mdp.QualitativeMDP):
    GRAPH_PROPERTY = mdp.QualitativeMDP.GRAPH_PROPERTY.copy()

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
        return list(itertools.product(self._game.actions(), self._game.sensor_query()))

    def init_state(self):
        initial_obs = self._game.init_observation()
        initial_state = (self._game.init_state(), tuple(initial_obs[0]), tuple(initial_obs[1]))
        return initial_state

    def final(self, state):
        st, b1, b2 = state

        if type(b1[1]) == int:
            b1 = b1
        else:
            b1 = list(b1)

        if type(b2[1]) == int:
            b2 = b2
        else:
            b2 = list(b2)

        p1_flag = 1
        p2_flag = 1

        if type(b1[1]) == int:
            if self._game.final(b1) == 0:
                p1_flag = 0
        else:
            for s in list(b1):
                if self._game.final(s) == 0:
                    p1_flag = 0
                    break

        if p1_flag == 0:
            return False

        if type(b2[1]) == int:
            if self._game.final(b2) == 0:
                p2_flag = 0
        else:
            for state in list(b2):
                if self._game.final(state) == 0:
                    p2_flag = 0
                    break

        if p1_flag == 1 and p2_flag == 0:
            return True
        else:
            return False

    def belief_one_dash(self, belief, action):
        a, X = action
        post_belief = set()

        if type(belief[1]) == int:
            post = self._game.delta(belief, a)
            post_belief = post_belief.union(set(post))
        else:
            for s in belief:
                post = self._game.delta(s, a)
                post_belief = post_belief.union(set(post))

        return list(post_belief)

    def belief_two_dash(self, belief):
        post_belief = set()
        for a in self._game.actions():
            if type(belief[1]) == int:
                post = self._game.delta(belief, a)
                post_belief = post_belief.union(set(post))
            else:
                for s in belief:
                    post = self._game.delta(s, a)
                    post_belief = post_belief.union(set(post))

        return list(post_belief)

    def delta(self, state, action):
        a, X = action
        delta_states = list()
        if not self.final(state):
            st, b1, b2 = state

            if type(b1[1]) == int:
                b1 = b1
            else:
                b1 = list(b1)

            if type(b2[1]) == int:
                b2 = b2
            else:
                b2 = list(b2)

            next_states = self._game.delta(st, a)
            post_b1 = self.belief_one_dash(b1, action)
            post_b2 = self.belief_two_dash(b2)

            for nx_st in next_states:
                observation_1, observation_2 = self._game.observation(nx_st, X)
                b1_dash = set(post_b1).intersection(set(observation_1))
                b2_dash = set(post_b2).intersection(set(observation_2))
                if len(b1_dash) == 1:
                    b1_dash = tuple(b1_dash)[0]
                else:
                    b1_dash = tuple(b1_dash)

                if len(b2_dash) == 1:
                    b2_dash = tuple(b2_dash)[0]
                else:
                    b2_dash = tuple(b2_dash)

                delta_states.append((nx_st, b1_dash, b2_dash))

        else:
            delta_states.append(state)

        return delta_states

    # PATCH. We need a way to store belief equivalence
    @models.register_property(GRAPH_PROPERTY)
    def belief_equivalent(self):
        states = self.__states
        equivalence_cls = {0: []}
        for (s, b1, b2), (t, c1, c2) in itertools.product(states, states):
            if b1 == c1 and b2 == c2:
                uid = states[(s, b1, b2)]
                vid = states[(t, c1, c2)]
                flag = 0
                fin_item = 0
                for items in equivalence_cls:
                    if uid in equivalence_cls[items] or vid in equivalence_cls[items]:
                        new_keys = equivalence_cls[items]
                        new_keys.append(uid)
                        new_keys.append(vid)
                        equivalence_cls[items] = new_keys
                        flag = 0
                        break
                    else:
                        flag = 1
                        fin_item = items

                if flag == 1:
                    equivalence_cls[fin_item + 1] = [uid, vid]

        return equivalence_cls
