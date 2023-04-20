import multiprocessing
import pathlib
import subprocess
import ggsolver
import ggsolver.ioutils as io
from ggsolver.logic.base import *
from loguru import logger
from dd.autoref import BDD


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

    def __init__(self, f_str=None, options=None, atoms=None):
        """
        Given an LTL formula, SpotAutomaton determines the best options for spot.translate() function
        to generate a deterministic automaton in ggsolver.Automaton format.
        :param f_str: (str) LTL formula.
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
        super(SpotAutomaton, self).__init__()

        # Instance variables
        self._f_str = f_str
        self._user_atoms = set(atoms) if atoms is not None else set()

        # If options are not given, determine the set of options to generate deterministic automaton with
        # state-based acceptance condition.
        if options is None:
            options = self._determine_options()

        logger.debug(f"[INFO] Translating {self._f_str} with options={options}.")
        self.spot_aut = spot.translate(f_str, *options)

        # Set the acceptance condition (in ggsolver terms)
        name = self.spot_aut.acc().name()
        if name == "Büchi" and spot.mp_class(f_str).upper() in ["S"]:
            self._acc_cond = (Automaton.ACC_BUCHI, 0)
        elif name == "Büchi" and spot.mp_class(f_str).upper() in ["B", "G"]:
            self._acc_cond = (Automaton.ACC_REACH, 0)
        elif name == "Büchi" and spot.mp_class(f_str).upper() in ["O", "R"]:
            self._acc_cond = (Automaton.ACC_BUCHI, 0)
        elif name == "co-Büchi":
            self._acc_cond = (Automaton.ACC_PARITY, 0)
        elif name == "all":
            self._acc_cond = (Automaton.ACC_SAFETY, 0)
        else:  # name contains "parity":
            self._acc_cond = (Automaton.ACC_PARITY, 0)

    def _determine_options(self):
        """
        Determines the options based on where the given LTL formula lies in Manna-Pnueli hierarchy.
        """
        mp_cls = spot.mp_class(self.formula())
        if mp_cls.upper() == "S":
            # return 'Monitor', "Deterministic", "High", "Complete", "Unambiguous", "SBAcc"
            return 'Buchi', "Deterministic", "High", "Complete", "Unambiguous", "SBAcc"
        elif mp_cls.upper() == "G" or mp_cls.upper() == "B" or mp_cls.upper() == "O" or mp_cls.upper() == "R":
            return 'Buchi', "Deterministic", "High", "Complete", "Unambiguous", "SBAcc"
        elif mp_cls.upper() == "P":
            # return 'coBuchi', "Deterministic", "High", "Complete", "Unambiguous", "SBAcc"
            return 'parity min even', "Deterministic", "High", "Complete", "Unambiguous", "SBAcc"
        else:  # cls.upper() == "T":
            return 'parity min even', "Deterministic", "High", "Complete", "Unambiguous", "SBAcc", "colored"

    def states(self):
        """ States of automaton. """
        return (f"q{i}" for i in range(self.spot_aut.num_states()))

    def atoms(self):
        """ Atomic propositions appearing in LTL formula. """
        return {str(ap) for ap in self.spot_aut.ap()} | self._user_atoms

    def inputs(self):
        util.apply_atoms_limit(self.atoms())
        return util.powerset(self.atoms())

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
        for t in self.spot_aut.out(int(state[1:])):
            label = spot.bdd_format_formula(bdd_dict, t.cond)
            label = spot.formula(label)
            if label.is_ff():
                continue
            elif label.is_tt():
                next_states.append(f"q{int(t.dst)}")
            else:
                label = spot.formula(label).to_str('spin')
                v = bdd.add_expr(label)
                if bdd.let(inp_dict, v) == bdd.true:
                    next_states.append(f"q{int(t.dst)}")

        # Return based on whether automaton is deterministic or non-deterministic.
        #   If automaton is deterministic but len(next_states) = 0, then automaton is incomplete, return None.
        if self.is_deterministic() and len(next_states) > 0:
            return next_states[0]

        if not self.is_deterministic():
            return next_states

    def init_state(self):
        """ Initial state of automaton. """
        return int(self.spot_aut.get_init_state_number())

    def final(self):
        """ Maps every state to its acceptance set. """
        if not self.is_state_based_acc():
            raise NotImplementedError
        return (self.spot_aut.state_acc_sets(int(state[1:])).sets() for state in self.states())

    def acc_cond(self):
        """
        Returns acceptance condition according to ggsolver definitions:
        See `ACC_REACH, ...` variables in Automaton class.
        See :meth:`SpotAutomaton.spot_acc_cond` for acceptance condition in spot's nomenclature.
        """
        return self._acc_cond[0]

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

    def is_semi_deterministic(self):
        """
        Is the automaton semi-deterministic?
        See https://spot.lrde.epita.fr/doxygen/namespacespot.html#a56b3f00b7b93deafb097cad595998783
        """
        return bool(spot.is_semi_deterministic(self.spot_aut))

    def acc_name(self):
        """ Name of acceptance condition as per spot's nomenclature. """
        return self.spot_aut.acc().name()

    def spot_acc_cond(self):
        """
        Acceptance condition in spot's nomenclature.
        """
        return str(self.spot_aut.get_acceptance())

    def formula(self):
        """ The LTL Formula. """
        return self._f_str

    def is_state_based_acc(self):
        """ Is the acceptance condition state-based? """
        return bool(self.spot_aut.prop_state_acc())

    def is_weak(self):
        """
        Are transitions of an SCC all belong to the same acceptance sets?
        """
        return bool(self.spot_aut.prop_weak())

    def is_inherently_weak(self):
        """ Is it the case that accepting and rejecting cycles cannot be mixed in the same SCC? """
        return bool(self.spot_aut.prop_inherently_weak())


class RabinizerAutomaton(Automaton):
    """
    Class uses `rabinizer4` tool to translate any LTL formula to Rabin automaton.

    Process:
    - Construct a shell command to translate input LTL formula string to Rabin automaton.
    - `rabinizer4` generates a transition-based automaton in `.hoa` format.
    - Use spot's `autfilt` tool to transform `.hoa` file to `.dot` file.
        In addition, this transformation also converts transition-based acceptance to state-based.
        Also, the Rabin automaton is completed, if incomplete.
    - Use `ggsolver.ioutils` to read `.dot` file into a graph.
    - Use the graph to construct RabinAutomaton instance.

    To use this class:
    >>> r = RabinizerAutomaton(f_str='GFa | FGb', fname="GFa")
    >>> r.delta("q0", {'a'})  # Gets next state if only atom "a" is true, equivalent to `a & !b`.

    The implementation works, but may not be stable.
    FIXME.
    """
    def __init__(self, f_str, **kwargs):
        """
        kwargs:
            - "fpath" (str, Path-like) Path of output HOA file generated by rabinizer.
            - "save_output" (bool). When true, the HOA file and DOT files are saved.
        """
        super(RabinizerAutomaton, self).__init__()
        self.f_str = f_str
        self._formula = spot.formula(f_str)
        self._atoms = set(self._collect_atoms()) | kwargs.get("atoms", set())

        self.directory = kwargs.get("directory", f"out")
        self.fname = kwargs.get("fname", "tmp_aut.hoa")
        if not pathlib.Path(self.directory).exists():
            pathlib.Path(self.directory).mkdir()
        self.fpath = os.path.join(self.directory, f"{self.fname}.hoa")
        if pathlib.Path(self.fpath).exists():
            pathlib.Path(self.fpath).unlink()
        self._cache_state2node = dict()

        # Run a command on the command line
        command = ['ltl2dra', '-O', f'{self.fpath}', '-c', f'{f_str}']
        print(command)
        result = subprocess.run(command, stdout=subprocess.PIPE)
        if result.returncode != 0:
            raise RuntimeError(f"Rabinizer failed while translating {f_str=}.")

        # Convert HOA to DOT for reading
        dot_path = os.path.join(self.directory, f"{self.fname}.dot")
        command = ['autfilt', '-F', f'{self.fpath}', '-S', '-d', '-o', f"{dot_path}"]
        print(' '.join(command))
        result = subprocess.run(command, stdout=subprocess.PIPE)
        if result.returncode != 0:
            print(result.stdout.decode('utf-8'))
            raise RuntimeError(f"autfilt failed while transform HOA to DOT.")

        # Construct automaton from dot file.
        graph = ggsolver.Graph()
        io.from_dot(dot_path, graph, backend="rabinizer")
        self.from_graph(graph)

    def _collect_atoms(self):
        atoms = set()

        def traversal(node: spot.formula, atoms_):
            if node.is_literal():
                if "!" not in node.to_str():
                    atoms_.add(node.to_str())
                    return True
            return False

        self._formula.traverse(traversal, atoms)
        return atoms

    def atoms(self):
        return self._atoms

    def inputs(self):
        pass

    def is_deterministic(self):
        return True

    def is_complete(self):
        return True

    def formula(self):
        return self._formula

    def acc_cond(self):
        return Automaton.ACC_RABIN

    def num_acc_sets(self):
        return len(self.final())

    def from_graph(self, graph):
        # Populate state to node cache
        self._cache_state2node = dict()
        for node in graph.nodes():
            self._cache_state2node[graph["state"][node]] = node

        if graph["acc_cond"] != self.acc_cond():
            raise TypeError(f"Cannot load automaton with {graph['acc_cond']=} into an automaton with {self.acc_cond()=}")

        def states_():
            return (graph["state"][uid] for uid in graph.nodes())

        def init_state_():
            return graph["init_state"]

        def final_():
            return graph["final"]

        def delta_(state, inp):
            uid = self._cache_state2node[state]
            if isinstance(inp, (list, set, tuple)) and len(inp) > 0:
                inp_str = " & ".join(inp)
                others = " & ".join([f"!{atom}" for atom in set(self.atoms()) - set(inp)])
                if len(others) > 0:
                    inp_str += " & " + others
            else:
                inp_str = " & ".join([f"!{atom}" for atom in self.atoms()])

            next_states = list()
            for _, vid, key in graph.out_edges(uid):
                f_str = graph["label"][uid, vid, key]
                if PL(formula=f"({inp_str}) -> {f_str}") == PL("true"):
                    next_states.append(graph["state"][vid])

            assert len(next_states) == 1
            return next_states[0]

        setattr(self, "states", states_)
        setattr(self, "init_state", init_state_)
        setattr(self, "final", final_)
        setattr(self, "delta", delta_)


class DFA(Automaton):
    """
    Represents a Deterministic Finite-state base.Automaton.
    - Acceptance Type: `base.Automaton.ACC_REACH`
    - Acceptance condition: `(Reach, 0)`
        - **Accepts:** Finite words.
        - **Interpretation:** :math:`\mathsf{Last}(\\rho) \in F` where :math:`F = \{q \in Q \mid \mathsf{AccSet}(q) = 0\}`
    - Number of Acceptance Sets: `1`
    - `final(state)` function returns either `-1` to indicate that the state is not accepting or `0` to
      indicate that the state is accepting with acceptance set `0`.
    """
    def __init__(self, states=None, atoms=None, trans_dict=None, init_state=None, final=None):
        """
        Constructs a DFA.
        :param states: (Iterable) An iterable over states in the automaton.
        :param atoms: (Iterable[str]) An iterable over atomic propositions in the automaton.
        :param trans_dict: (dict) A dictionary defining the (deterministic) transition function of automaton.
                      Format of dictionary: {state: {logic.PLFormula: state}}
        :param init_state: (object) The initial state, a member of states iterable.
        :param final: (Iterable[states]) The set of final states, a subset of states iterable.
        """
        super(DFA, self).__init__(states=states, atoms=atoms, trans_dict=trans_dict, init_state=init_state, final=final,
                                      is_deterministic=True,
                                      acc_cond=Automaton.ACC_REACH)

    @classmethod
    def intesection_product(cls, *args):
        if len(args) == 0:
            raise ValueError("Expect more than one DFA(s) to compute the product.")

        if len(args) == 1:
            return args[0]

        # Ensure all DFAs have same atoms

    @classmethod
    def union_product(cls, *args):
        pass


class DBA(Automaton):
    """
    Represents a Deterministic Finite-state base.Automaton.
    - Acceptance Type: `base.Automaton.ACC_REACH`
    - Acceptance condition: `(Reach, 0)`
        - **Accepts:** Finite words.
        - **Interpretation:** :math:`\mathsf{Last}(\\rho) \in F` where :math:`F = \{q \in Q \mid \mathsf{AccSet}(q) = 0\}`
    - Number of Acceptance Sets: `1`
    - `final(state)` function returns either `-1` to indicate that the state is not accepting or `0` to
      indicate that the state is accepting with acceptance set `0`.
    """
    def __init__(self, states=None, atoms=None, trans_dict=None, init_state=None, final=None):
        """
        Constructs a DFA.
        :param states: (Iterable) An iterable over states in the automaton.
        :param atoms: (Iterable[str]) An iterable over atomic propositions in the automaton.
        :param trans_dict: (dict) A dictionary defining the (deterministic) transition function of automaton.
                      Format of dictionary: {state: {logic.PLFormula: state}}
        :param init_state: (object) The initial state, a member of states iterable.
        :param final: (Iterable[states]) The set of final states, a subset of states iterable.
        """
        super(DBA, self).__init__(states=states, atoms=atoms, trans_dict=trans_dict, init_state=init_state, final=final,
                                  is_deterministic=True,
                                  acc_cond=Automaton.ACC_BUCHI)

    @classmethod
    def intesection_product(cls, *args):
        if len(args) == 0:
            raise ValueError("Expect more than one DFA(s) to compute the product.")

        if len(args) == 1:
            return args[0]

        # Ensure all DFAs have same atoms

    @classmethod
    def union_product(cls, *args):
        pass


class DRA(Automaton):
    """
    Represents a Deterministic Finite-state Rabin Automaton.
    - Acceptance Type: `base.Automaton.ACC_RABIN`
    - Acceptance condition: `(Rabin, 0)`
        - **Accepts:** Infinite words.
        - **Interpretation:** :math:`\mathsf{Last}(\\rho) \in F` where :math:`F = \{q \in Q \mid \mathsf{AccSet}(q) = 0\}`
    - Number of Acceptance Sets: `1`
    - `final(state)` function returns either `-1` to indicate that the state is not accepting or `0` to
      indicate that the state is accepting with acceptance set `0`.
    """
    def __init__(self, states=None, atoms=None, trans_dict=None, init_state=None, final=None):
        """
        Constructs a DFA.
        :param states: (Iterable) An iterable over states in the automaton.
        :param atoms: (Iterable[str]) An iterable over atomic propositions in the automaton.
        :param trans_dict: (dict) A dictionary defining the (deterministic) transition function of automaton.
                      Format of dictionary: {state: {logic.PLFormula: state}}
        :param init_state: (object) The initial state, a member of states iterable.
        :param final: (Iterable[states]) The set of final states, a subset of states iterable.
        """
        super(DRA, self).__init__(states=states, atoms=atoms, trans_dict=trans_dict, init_state=init_state, final=final,
                                  is_deterministic=True,
                                  acc_cond=Automaton.ACC_RABIN)

    @classmethod
    def intesection_product(cls, *args):
        if len(args) == 0:
            raise ValueError("Expect more than one DFA(s) to compute the product.")

        if len(args) == 1:
            return args[0]

        # Ensure all DFAs have same atoms

    @classmethod
    def union_product(cls, *args):
        pass


class DPA(Automaton):
    """
    Represents a Deterministic Finite-state Rabin Automaton.
    - Acceptance Type: `base.Automaton.ACC_RABIN`
    - Acceptance condition: `(Rabin, 0)`
        - **Accepts:** Infinite words.
        - **Interpretation:** :math:`\mathsf{Last}(\\rho) \in F` where :math:`F = \{q \in Q \mid \mathsf{AccSet}(q) = 0\}`
    - Number of Acceptance Sets: `1`
    - `final(state)` function returns either `-1` to indicate that the state is not accepting or `0` to
      indicate that the state is accepting with acceptance set `0`.
    """
    def __init__(self, states=None, atoms=None, trans_dict=None, init_state=None, final=None):
        """
        Constructs a DPA.
        :param states: (Iterable) An iterable over states in the automaton.
        :param atoms: (Iterable[str]) An iterable over atomic propositions in the automaton.
        :param trans_dict: (dict) A dictionary defining the (deterministic) transition function of automaton.
                      Format of dictionary: {state: {logic.PLFormula: state}}
        :param init_state: (object) The initial state, a member of states iterable.
        :param final: (Iterable[states]) The set of final states, a subset of states iterable.
        """
        super(DPA, self).__init__(states=states, atoms=atoms, trans_dict=trans_dict, init_state=init_state, final=final,
                                  is_deterministic=True,
                                  acc_cond=Automaton.ACC_PARITY)

    @classmethod
    def intesection_product(cls, *args):
        if len(args) == 0:
            raise ValueError("Expect more than one DFA(s) to compute the product.")

        if len(args) == 1:
            return args[0]

        # Ensure all DFAs have same atoms

    @classmethod
    def union_product(cls, *args):
        pass
