import pytest
import unittest
import sys
import ggsolver.logic as logic
from loguru import logger
logger.add(sys.stdout, level="DEBUG")


class TestLTLInstantiation(unittest.TestCase):
    def test_ltl_instantiation(self):
        _ = logic.LTL(formula="false")
        _ = logic.LTL(formula="true")
        _ = logic.LTL(formula="a")
        _ = logic.LTL(formula="a | b")
        _ = logic.LTL(formula="a & b")
        _ = logic.LTL(formula="a | (b & c)")
        _ = logic.LTL(formula="Fa | G(b & c)")
        _ = logic.LTL(formula="GFa & FG(b & c)")
        _ = logic.LTL(formula="F(a & Fb)")
        _ = logic.LTL(formula="GF(a & Fb)")

        with pytest.raises(SyntaxError):
            logic.LTL(formula="G")

    def test_scltl_instantiation(self):
        _ = logic.ScLTL(formula="false")
        _ = logic.ScLTL(formula="true")
        _ = logic.ScLTL(formula="a")
        _ = logic.ScLTL(formula="a | b")
        _ = logic.ScLTL(formula="a & b")
        _ = logic.ScLTL(formula="a | (b & c)")
        _ = logic.ScLTL(formula="F(a & Fb)")

        with pytest.raises(TypeError):
            _ = logic.ScLTL(formula="GF(a & Fb)")

        with pytest.raises(TypeError):
            _ = logic.ScLTL(formula="Fa | G(b & c)")

    def test_ltl_functionality(self):
        formula = logic.LTL("Fa & Gb")

        self.assertEqual({'a', 'b'}, set(formula.atoms()))
        self.assertEqual(True, formula.evaluate({'a', 'b'}))
        self.assertEqual(False, formula.evaluate({'a'}))

        self.assertEqual(logic.LTL("G(a & b)"), logic.LTL("Ga & Gb"))

        formula2 = logic.LTL("Fa & Gb", atoms={'a', 'b', 'c'})
        self.assertEqual({'a', 'b', 'c'}, set(formula2.atoms()))

        formula3 = logic.LTL("Fa & Gb", atoms={'c'})
        self.assertEqual({'a', 'b', 'c'}, set(formula3.atoms()))


class TestLTLTranslate(unittest.TestCase):
    def setUp(self):
        self.f_bottom = logic.ScLTL("p")
        self.f_guarantee = logic.ScLTL("Fp")
        self.f_safety = logic.LTL("Gp")
        self.f_obligation = logic.LTL("Fp & Gq")
        self.f_recurrence = logic.LTL("GFp")
        self.f_persistence = logic.LTL("FGp")
        self.f_reactivity = logic.LTL("FGp & GFq")

    def test_translate_bottom(self):
        dfa = logic.DFA().from_automaton(self.f_bottom.translate())
        self.assertEqual({"q0", "q1", "q2"}, set(dfa.states()))
        self.assertEqual({'p'}, set(dfa.atoms()))
        self.assertEqual("q0", dfa.delta("q1", {'p'}))
        self.assertEqual("q2", dfa.delta("q1", set()))

    def test_translate_guarantee(self):
        dfa = logic.DFA().from_automaton(self.f_guarantee.translate())
        self.assertEqual({"q0", "q1"}, set(dfa.states()))
        self.assertEqual({'p'}, set(dfa.atoms()))
        self.assertEqual("q0", dfa.delta("q1", {'p'}))
        self.assertEqual("q0", dfa.delta("q0", {'p'}))
        self.assertEqual("q0", dfa.delta("q0", {'!p'}))

    def test_translate_safety(self):
        with pytest.raises(TypeError):
            dfa = logic.DFA().from_automaton(self.f_safety.translate())
        dba = logic.DBA().from_automaton(self.f_safety.translate())

    def test_translate_obligation(self):
        with pytest.raises(TypeError):
            dfa = logic.DFA().from_automaton(self.f_obligation.translate())
        dba = logic.DBA().from_automaton(self.f_obligation.translate())

    def test_translate_recurrence(self):
        with pytest.raises(TypeError):
            dfa = logic.DFA().from_automaton(self.f_recurrence.translate())
        dba = logic.DBA().from_automaton(self.f_recurrence.translate())

    def test_translate_persistence(self):
        with pytest.raises(TypeError):
            dfa = logic.DFA().from_automaton(self.f_persistence.translate())
        dpa = logic.DPA().from_automaton(self.f_persistence.translate())

    def test_translate_reactivity(self):
        with pytest.raises(TypeError):
            dfa = logic.DFA().from_automaton(self.f_reactivity.translate())
        dpa = logic.DPA().from_automaton(self.f_reactivity.translate())

