from abc import ABC

import spot
import ggsolver.logic.base as base
from ggsolver.logic.formula import BaseFormula, ParsingError
from ggsolver.logic.automata import DFA
# import ggsolver.interfaces.i_spot as i_spot
# import ggsolver.logic.automata as automata
# from ggsolver.util import apply_atoms_limit, powerset


class LTL(BaseFormula):
    """
    LTL formula is internally represented as spot.formula instance.
    """
    __hash__ = BaseFormula.__hash__

    def __init__(self, f_str, atoms=None):
        super(LTL, self).__init__(f_str, atoms)
        self._repr = spot.formula(f_str)
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
        return base.SpotAutomaton(formula=self.f_str, atoms=self.atoms())

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


class ScLTL(LTL):
    """
    ScLTL formula is internally represented as spot.formula instance.
    """

    __hash__ = LTL.__hash__

    def __init__(self, f_str, atoms=None):
        super(ScLTL, self).__init__(f_str, atoms)
        mp_class = spot.mp_class(self._repr).upper()
        if mp_class not in ["B", "G"]:
            raise ParsingError(f"Given formula:{f_str} is not an ScLTL formula.")

    def __eq__(self, other: BaseFormula):
        try:
            return spot.are_equivalent(self.f_str, other.f_str)
        except Exception:
            return False

    def translate(self):
        aut = super(ScLTL, self).translate()
        dfa = DFA()
        dfa.from_automaton(aut)
        return dfa

    def substitute(self, subs_map=None):
        raise NotImplementedError("Will be implemented in future. ")
