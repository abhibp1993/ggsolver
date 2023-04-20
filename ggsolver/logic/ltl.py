from ggsolver.logic.base import *
from ggsolver.logic.automata import *


class LTL(BaseFormula):
    """
    LTL formula is internally represented as spot.formula instance.
    """
    __hash__ = BaseFormula.__hash__

    def __init__(self, formula, atoms=None):
        super(LTL, self).__init__(formula, atoms)
        self._repr = spot.formula(formula)
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
    def translate(self, backend="spot"):
        if backend == "rabinizer":
            return DRA().from_automaton(RabinizerAutomaton(f_str=self.f_str, atoms=self.atoms()))
        elif backend == "spot":
            aut = SpotAutomaton(f_str=self.f_str, atoms=self.atoms())
            if aut.acc_cond() == Automaton.ACC_PARITY:
                return DPA().from_automaton(aut)
            elif aut.acc_cond() == Automaton.ACC_BUCHI:
                return DBA().from_automaton(aut)
            elif aut.acc_cond() == Automaton.ACC_REACH:
                return DFA().from_automaton(aut)
            else:
                raise TypeError(f"LTL({self.f_str}) could not be translated using `spot` backend. "
                                f"Generated SpotAutomaton has {aut.acc_cond()=}.")
        else:
            raise ValueError(f"Unrecognized {backend=} for translation of {self.__class__.__name__} formula.")

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
        return self._atoms

    # ==================================================================
    # SPECIAL METHODS OF LTL CLASS
    # ==================================================================
    def simplify(self, f_str=None):
        """
        Simplifies a linear temporal logic (LTL) formula.
        We use the `boolean_to_isop=True` option for `spot.simplify`.
        See https://spot.lrde.epita.fr/doxygen/classspot_1_1tl__simplifier__options.html
        :param f_str: (str or None) Input formula string. If not provided, self._user_string is used.
        :return: (str) String representing simplified formula.
        """
        if f_str is None:
            return spot.simplify(self._repr, boolean_to_isop=True).to_str()

        # return spot.simplify(self._repr, boolean_to_isop=True).to_str()
        return spot.simplify(
            spot.formula(f_str),
            boolean_to_isop=True
        ).to_str()


class ScLTL(LTL):
    """
    ScLTL formula is internally represented as spot.formula instance.
    """

    __hash__ = LTL.__hash__

    def __init__(self, formula, atoms=None):
        super(ScLTL, self).__init__(formula, atoms)
        mp_class = spot.mp_class(self._repr).upper()
        if mp_class not in ["B", "G"]:
            raise TypeError(f"Given formula:{formula} is not an ScLTL formula.")

    def __eq__(self, other: BaseFormula):
        return spot.are_equivalent(self.f_str, other.f_str)

    def translate(self, backend="spot"):
        aut = super(ScLTL, self).translate(backend=backend)
        dfa = DFA()
        dfa.from_automaton(aut)
        return dfa
