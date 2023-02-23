import itertools
import logging
import spot
from dd.autoref import BDD
from ggsolver.logic.formula import BaseFormula, ParsingError
from tqdm import tqdm

import ggsolver.util as util
import ggsolver.models as models


class PL(BaseFormula):
    """
    PL formula is internally represented as spot.formula instance.
    """
    def __init__(self, f_str, atoms=None):
        super(PL, self).__init__(f_str, atoms)
        self._repr = spot.formula(f_str)
        if not self._repr.is_boolean():
            raise ParsingError(f"Given formula:{f_str} is not a PL formula.")
        self._atoms = self._collect_atoms()

    def __str__(self):
        return str(self.f_str)

    def __eq__(self, other: BaseFormula):
        try:
            return spot.are_equivalent(self.f_str, other.f_str)
        except Exception:
            return False

    def _collect_atoms(self):
        atoms = set()

        def traversal(node: spot.formula, atoms_):
            if node.is_literal():
                if "!" not in node.to_str():
                    atoms_.add(node.to_str())
                    return True
            return False

        self._repr.traverse(traversal, atoms)
        return self._atoms | atoms

    # ==================================================================
    # IMPLEMENTATION OF ABSTRACT METHODS
    # ==================================================================
    def translate(self):
        """
        Translate a propositional logic formula to an automaton.
        :return: (:class:`SpotAutomaton`) SpotAutomaton representing the automaton for PL formula.
        """
        return SpotAutomaton(formula=self.f_str, atoms=self.atoms())

    def substitute(self, subs_map=None):
        raise NotImplementedError("To be implemented in future.")

    def evaluate(self, true_atoms):
        """
        Evaluates a propositional logic formula given the set of true atoms.

        :param true_atoms: (Iterable[str]) A propositional logic formula.
        :return: (bool) True if formula is true, otherwise False.
        """
        # Define a transform to apply to AST of spot.formula.
        def transform(node: spot.formula):
            if node.is_literal():
                if "!" not in node.to_str():
                    if node.to_str() in true_atoms:
                        return spot.formula.tt()
                    else:
                        return spot.formula.ff()

            return node.map(transform)

        # Apply the transform and return the result.
        # Since every literal is replaced by true or false,
        #   the transformed formula is guaranteed to be either true or false.
        return True if transform(self._repr).is_tt() else False

    def atoms(self):
        """
        Gets the list of atoms associated with PL formula.

        The list may contain atoms that do not appear in the formula, if the user has provided it.
        :return: (List[str]) List of atoms.
        """
        return self._atoms

    # ==================================================================
    # SPECIAL METHODS OF PL CLASS
    # ==================================================================
    def simplify(self):
        """
        Simplifies a propositional logic formula.

        We use the `boolean_to_isop=True` option for `spot.simplify`.
        See https://spot.lrde.epita.fr/doxygen/classspot_1_1tl__simplifier__options.html

        :return: (str) String representing simplified formula.
        """
        return spot.simplify(self._repr, boolean_to_isop=True).to_str()

    def allsat(self):
        """
        Generates the set of all satisfying assignments to atoms of the given propositional logic formula.

        .. note:: Complexity: Exponential in the number of atoms.
        """
        # Apply limitation on atoms we allow in ggsolver. Raises ValueError if |atoms| exceeds limit.
        util.apply_atoms_limit(self.atoms())

        # For each assignment, check whether the formula evaluates to True.
        # If yes, include it in set of all satisfying assignments.
        sat_assignments = []
        for assignment in util.powerset(self.atoms()):
            if self.evaluate(assignment):
                sat_assignments.append(assignment)
        return sat_assignments


class Automaton(models.GraphicalModel):
    """
    Represents an Automaton.

    .. math::
        \\mathcal{A} = (Q, \\Sigma := 2^{AP}, \\delta, q_0, F)

    In the `Automaton` class, each component is represented as a function.

    - The set of states :math:`Q` is represented by `Automaton.states` function,
    - The set of atomic propositions :math:`AP` is represented by `Automaton.atoms` function,
    - The set of symbols :math:`\\Sigma` is represented by `Automaton.sigma` function,
    - The transition function :math:`\\delta` is represented by `Automaton.delta` function,
    - The initial state :math:`q_0` is represented by `Automaton.init_state` function.

    An automaton may have one of the following acceptance conditions:

    - (:class:`Automaton.ACC_REACH`, 0)
    - (:class:`Automaton.ACC_SAFETY`, 0)
    - (:class:`Automaton.ACC_BUCHI`, 0)
    - (:class:`Automaton.ACC_COBUCHI`, 0)
    - (:class:`Automaton.ACC_PARITY`, 0)
    - (:class:`Automaton.ACC_PREF_LAST`, None)
    - (:class:`Automaton.ACC_ACC_PREF_MP`, None)

    """
    NODE_PROPERTY = models.GraphicalModel.NODE_PROPERTY.copy()
    EDGE_PROPERTY = models.GraphicalModel.EDGE_PROPERTY.copy()
    GRAPH_PROPERTY = models.GraphicalModel.GRAPH_PROPERTY.copy()

    ACC_REACH = "Reach"                                 #:
    ACC_SAFETY = "Safety"                               #:
    ACC_BUCHI = "Buchi"                                 #:
    ACC_COBUCHI = "co-Buchi"                            #:
    ACC_PARITY = "Parity Min Even"                      #:
    ACC_PREF_LAST = "Preference Last"                   #:
    ACC_PREF_MP = "Preference MostPreferred"            #:
    ACC_UNDEFINED = "undefined"
    ACC_TYPES = [
        ACC_UNDEFINED,
        ACC_REACH,
        ACC_SAFETY,
        ACC_BUCHI,
        ACC_COBUCHI,
        ACC_PARITY,
        ACC_PREF_LAST,
        ACC_PREF_MP
    ]                                   #: Acceptance conditions supported by Automaton.

    def __init__(self, **kwargs):
        """
        Supported keyword arguments:

        :param states: (Iterable) An iterable over states in the automaton.
        :param atoms: (Iterable[str]) An iterable over atomic propositions in the automaton.
        :param trans_dict: (dict) A dictionary defining the (deterministic) transition function of automaton.
                      Format of dictionary: {state: {logic.PLFormula: state}}
        :param init_state: (object) The initial state, a member of states iterable.
        :param final: (Iterable[states]) The set of final states, a subset of states iterable.
        :param acc_cond: (tuple) A tuple of automaton acceptance type and an acceptance set.
            For example, DFA has an acceptance condition of `(Automaton.ACC_REACH, 0)`.
        :param is_deterministic: (bool) Whether the Automaton is deterministic.
        """
        kwargs["input_domain"] = "atoms" if "input_domain" not in kwargs else kwargs["input_domain"]
        super(Automaton, self).__init__(**kwargs)

        # Process keyword arguments
        if "states" in kwargs:
            def states_():
                return list(kwargs["states"])
            self.states = states_

        if "atoms" in kwargs:
            def atoms_():
                return list(kwargs["atoms"])
            self.atoms = atoms_

        if "trans_dict" in kwargs:
            def delta_(state, inp):
                next_states = set()
                for formula, n_state in kwargs["trans_dict"][state].items():
                    if PL(f_str=formula, atoms=self.atoms()).evaluate(inp):
                        next_states.add(n_state)

                if self.is_deterministic():
                    if len(next_states) > 1:
                        raise ValueError("Non-determinism detected in a deterministic automaton. " +
                                         f"delta({state}, {inp}) -> {next_states}.")
                    return next(iter(next_states), None) if len(next_states) == 1 else None

                return next_states

            self.delta = delta_

        if "init_state" in kwargs:
            self.initialize(kwargs["init_state"])

        if "final" in kwargs:
            def final_(state):
                return [0] if state in kwargs["final"] else [-1]
            self.final = final_

        if "acc_cond" in kwargs:
            def acc_cond_():
                return kwargs["acc_cond"]
            self.acc_cond = acc_cond_

        if "is_deterministic" in kwargs:
            def is_deterministic_():
                return kwargs["is_deterministic"]
            self.is_deterministic = is_deterministic_

    # ==========================================================================
    # PRIVATE FUNCTIONS
    # ==========================================================================
    def _gen_underlying_graph_unpointed(self, graph):
        """
        Programmer's notes:
        1. Caches states (returned by `self.states()`) in self.__states variable.
        2. Assumes all states to be hashable.
        3. Parallel edges are merged using ORing of PL Formulas.
        """
        # Get states
        states = getattr(self, "states")
        states = list(states())

        # Add states to graph
        node_ids = list(graph.add_nodes(len(states)))

        # Cache states as a dictionary {state: uid}
        self.__states = dict(zip(states, node_ids))

        # Node property: state
        np_state = graph.NodePropertyMap(graph=graph)
        np_state.update(dict(zip(node_ids, states)))
        graph["state"] = np_state

        # Logging and printing
        logging.info(util.ColoredMsg.ok(f"[INFO] Processed node property: states. Added {len(node_ids)} states. [OK]"))

        # Get input function
        #   Specialized for automaton class: we expect input function to be atoms.
        assert self._input_domain == "atoms", "For automaton class, we expect input domain to be `atoms`. " \
                                              f"Currently it is set to '{self._input_domain}'."
        input_func = getattr(self, self._input_domain)
        atoms = input_func()
        inputs = util.powerset(atoms)
        logging.info(util.ColoredMsg.ok(f"[INFO] Input domain function detected as '{self._input_domain}'. [OK]"))

        # Graph property: input domain (stores the name of edge property that represents inputs)
        graph["input_domain"] = self._input_domain
        logging.info(util.ColoredMsg.ok(f"[INFO] Processed graph property: input_domain. [OK]"))

        # # Get input domain
        # inputs = input_func()

        # Edge properties: input, prob,
        ep_input = graph.EdgePropertyMap(graph=graph)
        ep_prob = graph.EdgePropertyMap(graph=graph, default=None)

        # Generate edges
        delta = getattr(self, "delta")
        edges = {uid: dict() for uid in node_ids}
        for state, inp in tqdm(itertools.product(self.__states.keys(), inputs),
                               total=len(self.__states) * 2 ** len(atoms),
                               desc="Specialized unpointed graphify adding edges for automaton "):

            new_edges = self._gen_edges(delta, state, inp)

            # Update graph edges
            uid = self.__states[state]
            for _, t, _, _ in new_edges:
                vid = self.__states[t]
                if vid not in edges[uid]:
                    edges[uid][vid] = list()
                edges[uid][vid].append(inp)

        for uid in edges.keys():
            for vid in edges[uid].keys():
                key = graph.add_edge(uid, vid)
                ep_input[uid, vid, key] = sat2formula(atoms, edges[uid][vid])
                ep_prob[uid, vid, key] = None

        # Add edge properties to graph
        graph["input"] = ep_input
        graph["prob"] = ep_prob
        logging.info(util.ColoredMsg.ok(f"[INFO] Processed edge property: input. [OK]"))
        logging.info(util.ColoredMsg.ok(f"[INFO] Processed graph property: prob. [OK]"))

    def _gen_underlying_graph_pointed(self, graph):
        raise NotImplementedError("Pointed graphify is not defined for automaton.")

    # ==========================================================================
    # FUNCTIONS TO BE IMPLEMENTED BY USER.
    # ==========================================================================
    @models.register_property(GRAPH_PROPERTY)
    def atoms(self):
        """
        Returns a list/tuple of atomic propositions.

        :return: (list of str) A list of atomic proposition.
        """
        raise NotImplementedError(f"{self.__class__.__name__}.atoms() is not implemented.")

    @models.register_property(NODE_PROPERTY)
    def final(self, state):
        """
        Returns the acceptance set associated with the given state.

        :param state: (an element of `self.states()`) A valid state.
        :return: (int) Acceptance set associated with the given state.
        """
        raise NotImplementedError(f"{self.__class__.__name__}.final() is not implemented.")

    @models.register_property(GRAPH_PROPERTY)
    def acc_type(self):
        """
        Acceptance type of the automaton.

        :return: A value from :class:`Automaton.ACC_TYPES`.
        """
        return self.acc_cond()[0]

    @models.register_property(GRAPH_PROPERTY)
    def acc_cond(self):
        """
        Acceptance condition of the automaton.

        :return: (2-tuple) A value of type (acc_type, acc_set) where acc_type is from :class:`Automaton.ACC_TYPES`
                 and acc_set is either an integer or a list of integer.
        """
        return self.ACC_UNDEFINED, None

    @models.register_property(GRAPH_PROPERTY)
    def num_acc_sets(self):
        """
        Number of acceptance sets.
        """
        raise NotImplementedError(f"{self.__class__.__name__}.num_acc_sets() is not implemented.")

    @models.register_property(GRAPH_PROPERTY)
    def is_complete(self):
        """
        Is the automaton complete? That is, is transition function well-defined at every state for any
        input symbol?
        """
        raise NotImplementedError

    # ==========================================================================
    # FUNCTIONS TO BE IMPLEMENTED BY USER.
    # ==========================================================================
    def sigma(self):
        """
        Returns the set of alphabet of automaton. It is the powerset of atoms().
        """
        return list(util.powerset(self.atoms()))

    def from_automaton(self, aut: 'Automaton'):
        """
        Constructs an Automaton from another Automaton instance.
        The input automaton's acceptance condition must match that of a current Automaton.
        """
        assert aut.acc_cond() == self.acc_cond(), f"aut.acc_cond(): {aut.acc_cond()}, self.acc_cond(): {self.acc_cond()}"

        # Copy all functions from automaton.
        self.states = aut.states
        self.delta = aut.delta
        self._input_domain = "atoms"

        for gp in aut.GRAPH_PROPERTY:
            setattr(self, gp, getattr(aut, gp))

        for np in aut.NODE_PROPERTY:
            setattr(self, np, getattr(aut, np))

        for ep in aut.EDGE_PROPERTY:
            setattr(self, ep, getattr(aut, ep))


class SpotAutomaton(Automaton):
    """
    `SpotAutomaton` constructs an :class:`Automaton` from an LTL specification string using
    `spot` (https://spot.lrde.epita.fr/) with customizations for `ggsolver`.

    **Customizations:** Since `ggsolver` contains several algorithms for reactive/controller synthesis,
    we prefer to construct deterministic automata. Given an LTL formula, `SpotAutomaton` automatically
    determines the best acceptance condition that would result in a deterministic automaton..

    Programmer's note: The graphified version of automaton does not use PL formulas as edge labels.
    This is intentionally done to be able to run our codes on robots that may not have logic libraries installed.
    """

    def __init__(self, formula=None, options=None, atoms=None):
        """
        Given an LTL formula, SpotAutomaton determines the best options for spot.translate() function
        to generate a deterministic automaton in ggsolver.Automaton format.

        :param formula: (str) LTL formula.
        :param options: (List/Tuple of str) Valid options for spot.translate() function. By default, the
            value is `None`, in which case, the options are determined automatically. See description below.

        **Default translation options:** While constructing an automaton using `spot`, we use the following
        options: `deterministic, high, complete, unambiguous, SBAcc`. If selected acceptance condition
        is parity, then we use `colored` option as well.

        The default options can be overriden. For quick reference, the following description is copied from
        `spot` documentation (spot.lrde.epita.fr/doxygen).

        The optional arguments should be strings among the following:
        - at most one in 'GeneralizedBuchi', 'Buchi', or 'Monitor',
        'generic', 'parity', 'parity min odd', 'parity min even',
        'parity max odd', 'parity max even', 'coBuchi'
        (type of acceptance condition to build)

        - at most one in 'Small', 'Deterministic', 'Any'
          (preferred characteristics of the produced automaton)
        - at most one in 'Low', 'Medium', 'High'
          (optimization level)
        - any combination of 'Complete', 'Unambiguous',
          'StateBasedAcceptance' (or 'SBAcc' for short), and
          'Colored' (only for parity acceptance)
        """
        # Construct the automaton
        super(SpotAutomaton, self).__init__(input_domain="atoms")

        # Instance variables
        self._formula = formula
        self._user_atoms = set(atoms) if atoms is not None else set()

        # If options are not given, determine the set of options to generate deterministic automaton with
        # state-based acceptance condition.
        if options is None:
            options = self._determine_options()

        print(f"[INFO] Translating {self._formula} with options={options}.")
        self.spot_aut = spot.translate(formula, *options)

        # Set the acceptance condition (in ggsolver terms)
        name = self.spot_aut.acc().name()
        if name == "Büchi" and spot.mp_class(formula).upper() in ["B", "S"]:
            self._acc_cond = (Automaton.ACC_SAFETY, 0)
        elif name == "Büchi" and spot.mp_class(formula).upper() in ["G"]:
            self._acc_cond = (Automaton.ACC_REACH, 0)
        elif name == "Büchi" and spot.mp_class(formula).upper() in ["O", "R"]:
            self._acc_cond = (Automaton.ACC_BUCHI, 0)
        elif name == "co-Büchi":
            self._acc_cond = (Automaton.ACC_COBUCHI, 0)
        elif name == "all":
            self._acc_cond = (Automaton.ACC_SAFETY, 0)
        else:  # name contains "parity":
            self._acc_cond = (Automaton.ACC_PARITY, 0)

    def _determine_options(self):
        """
        Determines the options based on where the given LTL formula lies in Manna-Pnueli hierarchy.
        """
        mp_cls = spot.mp_class(self.formula())
        if mp_cls.upper() == "B" or mp_cls.upper() == "S":
            return 'Monitor', "Deterministic", "High", "Complete", "Unambiguous", "SBAcc"
        elif mp_cls.upper() == "G" or mp_cls.upper() == "O" or mp_cls.upper() == "R":
            return 'Buchi', "Deterministic", "High", "Complete", "Unambiguous", "SBAcc"
        elif mp_cls.upper() == "P":
            return 'coBuchi', "Deterministic", "High", "Complete", "Unambiguous", "SBAcc"
        else:  # cls.upper() == "T":
            return 'parity min even', "Deterministic", "High", "Complete", "Unambiguous", "SBAcc", "colored"

    def states(self):
        """ States of automaton. """
        return list(range(self.spot_aut.num_states()))

    def atoms(self):
        """ Atomic propositions appearing in LTL formula. """
        return list({str(ap) for ap in self.spot_aut.ap()} | self._user_atoms)

    def delta(self, state, inp):
        """
        Transition function of automaton. For a deterministic automaton, returns a single state. Otherwise,
        returns a list/tuple of states.

        :param state: (object) A valid state.
        :param inp: (list) List of atoms that are true (an element of sigma).
        """
        # Preprocess inputs
        inp_dict = {p: True for p in inp} | {p: False for p in self.atoms() if p not in inp}

        # Initialize a BDD over set of atoms.
        bdd = BDD()
        bdd.declare(*self.atoms())

        # Get spot BDD dict to extract formula
        bdd_dict = self.spot_aut.get_dict()

        # Get next states
        next_states = []
        for t in self.spot_aut.out(state):
            label = spot.bdd_format_formula(bdd_dict, t.cond)
            label = spot.formula(label)
            if label.is_ff():
                continue
            elif label.is_tt():
                next_states.append(int(t.dst))
            else:
                label = spot.formula(label).to_str('spin')
                v = bdd.add_expr(label)
                if bdd.let(inp_dict, v) == bdd.true:
                    next_states.append(int(t.dst))

        # Return based on whether automaton is deterministic or non-deterministic.
        #   If automaton is deterministic but len(next_states) = 0, then automaton is incomplete, return None.
        if self.is_deterministic() and len(next_states) > 0:
            return next_states[0]

        if not self.is_deterministic():
            return next_states

    def init_state(self):
        """ Initial state of automaton. """
        return int(self.spot_aut.get_init_state_number())

    def final(self, state):
        """ Maps every state to its acceptance set. """
        if not self.is_state_based_acc():
            raise NotImplementedError
        return list(self.spot_aut.state_acc_sets(state).sets())

    def acc_cond(self):
        """
        Returns acceptance condition according to ggsolver definitions:
        See `ACC_REACH, ...` variables in Automaton class.
        See :meth:`SpotAutomaton.spot_acc_cond` for acceptance condition in spot's nomenclature.
        """
        return self._acc_cond

    def num_acc_sets(self):
        """ Number of acceptance sets. """
        return self.spot_aut.num_sets()

    def is_deterministic(self):
        """ Is the automaton deterministic? """
        return bool(self.spot_aut.prop_universal() and self.spot_aut.is_existential())

    def is_unambiguous(self):
        """
        There is at most one run accepting a word (but it might be recognized several time).
        See https://spot.lrde.epita.fr/concepts.html.
        """
        return bool(self.spot_aut.prop_unambiguous())

    def is_terminal(self):
        """
        Automaton is weak, accepting SCCs are complete, accepting edges may not go to rejecting SCCs.
        An automaton is weak if the transitions of an SCC all belong to the same acceptance sets.

        See https://spot.lrde.epita.fr/concepts.html
        """
        return bool(self.spot_aut.prop_terminal())

    def is_stutter_invariant(self):
        """
        The property recognized by the automaton is stutter-invariant
        (see https://www.lrde.epita.fr/~adl/dl/adl/michaud.15.spin.pdf)
        """
        return bool(self.spot_aut.prop_stutter_invariant())

    def is_complete(self):
        """ Is the automaton complete? """
        return bool(spot.is_complete(self.spot_aut))

    @models.register_property(Automaton.GRAPH_PROPERTY)
    def is_semi_deterministic(self):
        """
        Is the automaton semi-deterministic?
        See https://spot.lrde.epita.fr/doxygen/namespacespot.html#a56b3f00b7b93deafb097cad595998783
        """
        return bool(spot.is_semi_deterministic(self.spot_aut))

    @models.register_property(Automaton.GRAPH_PROPERTY)
    def acc_name(self):
        """ Name of acceptance condition as per spot's nomenclature. """
        return self.spot_aut.acc().name()

    @models.register_property(Automaton.GRAPH_PROPERTY)
    def spot_acc_cond(self):
        """
        Acceptance condition in spot's nomenclature.
        """
        return str(self.spot_aut.get_acceptance())

    @models.register_property(Automaton.GRAPH_PROPERTY)
    def formula(self):
        """ The LTL Formula. """
        return self._formula

    @models.register_property(Automaton.GRAPH_PROPERTY)
    def is_state_based_acc(self):
        """ Is the acceptance condition state-based? """
        return bool(self.spot_aut.prop_state_acc())

    @models.register_property(Automaton.GRAPH_PROPERTY)
    def is_weak(self):
        """
        Are transitions of an SCC all belong to the same acceptance sets?
        """
        return bool(self.spot_aut.prop_weak())

    @models.register_property(Automaton.GRAPH_PROPERTY)
    def is_inherently_weak(self):
        """ Is it the case that accepting and rejecting cycles cannot be mixed in the same SCC? """
        return bool(self.spot_aut.prop_inherently_weak())


def sat2formula(atoms, sat_assignments):
    """
    Given a subset of elements from powerset(atoms), generates a propositional logic formula
    that accepts exactly those elements.

    :param atoms: (Iterable[str]) The set of atoms.
    :param sat_assignments: (Iterable[powerset(atoms)]) A subset of powerset(atoms) representing
                            satisfiable assignments of the formula to be generated.
    :return: (str) String representing PL formula that accepts exactly the satisfying assignments.
    """
    # Generate all clauses
    formula = []
    for assignment in sat_assignments:
        # Each clause includes an ANDing of atoms in assignment and ANDing of negation of atoms not in assignment
        complete_acc = [p if p in assignment else f"!{p}" for p in atoms]
        formula.append(f"({' & '.join(complete_acc)})")

    # Construct DNF formula by joining all clauses using disjunction
    formula = " | ".join(formula)
    formula = PL(f_str=formula, atoms=atoms).simplify()

    # Simplify the formula
    return PL(f_str=formula, atoms=atoms)
