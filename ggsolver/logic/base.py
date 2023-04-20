import inspect
import os
from abc import ABC, abstractmethod
import spot
import ggsolver.util as util

# Global Variable
PARSERS_DIR = os.path.dirname(inspect.getfile(inspect.currentframe()))
PARSERS_DIR = os.path.join(PARSERS_DIR, "grammars")


class ParsingError(ValueError):
    pass


class BaseFormula(ABC):
    def __init__(self, formula, atoms=None):
        self.user_str = formula
        self.f_str = self.simplify(formula)
        self._atoms = set(atoms) if atoms is not None else set()

    def __str__(self):
        return str(self.f_str)

    def __repr__(self):
        return f"{self.__class__.__name__}({self.f_str})"

    def __hash__(self):
        return hash(self.f_str)

    def __eq__(self, other):
        raise NotImplementedError("Marked abstract.")

    def update_atoms(self, atoms):
        self._atoms |= set(atoms)

    def simplify(self, f_str=None):
        """
        Simplifies the input formula.
        :param f_str: (str or None) Input formula string. If not provided, self._user_string is used.
        :return: (str) Simplified formula string.
        """
        if f_str is None:
            return self.user_str
        else:
            return f_str

    @abstractmethod
    def translate(self):
        pass

    @abstractmethod
    def evaluate(self, true_atoms):
        pass

    @abstractmethod
    def atoms(self):
        pass


class PL(BaseFormula):
    """
    PL formula is internally represented as spot.formula instance.
    """

    def __init__(self, formula, atoms=None):
        super(PL, self).__init__(formula, atoms)
        self._repr = spot.formula(formula)
        if not self._repr.is_boolean():
            raise ParsingError(f"Given formula:{formula} is not a PL formula.")
        self._atoms = self._collect_atoms()

    def __repr__(self):
        return f"{self.__class__.__name__}({self.f_str})"

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
        raise NotImplementedError
        # return SpotAutomaton(formula=self.f_str, atoms=self.atoms())

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
    def simplify(self, f_str=None):
        """
        Simplifies a propositional logic formula.
        We use the `boolean_to_isop=True` option for `spot.simplify`.
        See https://spot.lrde.epita.fr/doxygen/classspot_1_1tl__simplifier__options.html
        :param f_str: (str or None) Input formula string. If not provided, self._user_string is used.
        :return: (str) String representing simplified formula.
        """
        if f_str is None:
            return spot.simplify(self._repr, boolean_to_isop=True).to_str()

        return spot.simplify(
            spot.formula(f_str),
            boolean_to_isop=True
        ).to_str()

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


class Automaton:
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
    ACC_REACH = "Reach"  #: Reachability condition
    ACC_SAFETY = "Safety"  #: Safety condition
    ACC_BUCHI = "Buchi"  #: Buchi condition
    ACC_COBUCHI = "co-Buchi"  #: co-Buchi condition
    ACC_RABIN = "Rabin"  #: Rabin condition
    ACC_PARITY = "Parity Min Even"  #: Parity Min Even condition
    ACC_PREF_LAST = "Preference Last"  #:
    ACC_PREF_MP = "Preference MostPreferred"  #:
    ACC_UNDEFINED = "undefined"  #: Undefined acceptance condition
    ACC_TYPES = [
        ACC_UNDEFINED,
        ACC_REACH,
        ACC_SAFETY,
        ACC_BUCHI,
        ACC_COBUCHI,
        ACC_PARITY,
        ACC_PREF_LAST,
        ACC_PREF_MP
    ]  #: Acceptance conditions supported by Automaton.

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
        states = kwargs.get("states", None)
        atoms = kwargs.get("atoms", None)
        inputs = kwargs.get("inputs", None)
        trans_dict = kwargs.get("trans_dict", None)
        init_state = kwargs.get("init_state", None)
        final = kwargs.get("final", None)
        acc_cond = kwargs.get("acc_cond", None)
        num_acc_sets = kwargs.get("num_acc_sets", None)
        is_deterministic = kwargs.get("is_deterministic", None)
        is_complete = kwargs.get("is_complete", None)
        formula = kwargs.get("formula", None)

        if states is not None:
            def states_():
                return states
            setattr(self, "states", states_)

        if atoms is not None:
            def atoms_():
                return atoms
            setattr(self, "atoms", atoms_)

        if inputs is not None:
            def inputs_():
                return inputs
            setattr(self, "inputs", inputs_)

        if trans_dict is not None:
            def delta_(state, inp):
                if isinstance(inp, (list, set, tuple)):
                    inp = PL(" & ".join(inp))
                assert isinstance(inp, PL)

                next_states = list()
                for f_str, n_state in trans_dict[state].items():
                    if PL(formula=f"({inp}) -> {f_str}").simplify() == PL("true"):
                        next_states.append(n_state)
            setattr(self, "delta", delta_)

        if init_state is not None:
            def init_state_():
                return init_state
            setattr(self, "init_state", init_state_)

        if final is not None:
            def final_():
                return final
            setattr(self, "final", final_)

        if acc_cond is not None:
            def acc_cond_():
                return acc_cond
            setattr(self, "acc_cond", acc_cond_)

        if num_acc_sets is not None:
            def num_acc_sets_():
                return num_acc_sets
            setattr(self, "num_acc_sets", num_acc_sets_)

        if is_deterministic is not None:
            def is_deterministic_():
                return is_deterministic
            setattr(self, "is_deterministic", is_deterministic_)

        if is_complete is not None:
            def is_complete_():
                return is_complete
            setattr(self, "is_complete", is_complete_)

        if formula is not None:
            def formula_():
                return formula
            setattr(self, "formula", formula_)

    def states(self):
        raise NotImplementedError

    def atoms(self):
        raise NotImplementedError

    def inputs(self):
        raise NotImplementedError

    def delta(self, state, inp):
        raise NotImplementedError

    def init_state(self):
        raise NotImplementedError

    def final(self):
        raise NotImplementedError

    def acc_cond(self):
        raise NotImplementedError

    def num_acc_sets(self):
        raise NotImplementedError

    def is_deterministic(self):
        raise NotImplementedError

    def is_complete(self):
        raise NotImplementedError

    def formula(self):
        pass

    def from_graph(self, graph):
        raise NotImplementedError

    def graphify(self):
        pass

    def from_automaton(self, aut: 'Automaton'):
        """
        Constructs an Automaton from another Automaton instance.
        The input automaton's acceptance condition must match that of a current Automaton.
        """
        if aut.acc_cond() != self.acc_cond():
            raise TypeError(f"Cannot load automaton with {aut.acc_cond()=} into an automaton with {self.acc_cond()=}")

        # Copy all functions from automaton.
        setattr(self, "states", aut.states)
        setattr(self, "atoms", aut.atoms)
        setattr(self, "inputs", aut.inputs)
        setattr(self, "delta", aut.delta)
        setattr(self, "init_state", aut.init_state)
        setattr(self, "final", aut.final)
        setattr(self, "acc_cond", aut.acc_cond)
        setattr(self, "num_acc_sets", aut.num_acc_sets)
        setattr(self, "is_deterministic", aut.is_deterministic)
        setattr(self, "is_complete", aut.is_complete)
        setattr(self, "formula", aut.formula)
        return self

    def serialize(self):
        pass

    def deserialize(self, obj_dict):
        pass

    def to_hoa(self):
        pass

    def to_dot(self):
        pass
