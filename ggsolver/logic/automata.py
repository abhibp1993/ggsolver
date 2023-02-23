import ggsolver.logic.base as base


def filter_kwargs(states=None, atoms=None, trans_dict=None, init_state=None, final=None):
    kwargs = dict()
    if states is not None:
        kwargs["states"] = states

    if atoms is not None:
        kwargs["atoms"] = atoms

    if trans_dict is not None:
        kwargs["trans_dict"] = trans_dict

    if init_state is not None:
        kwargs["init_state"] = init_state

    if final is not None:
        kwargs["final"] = final

    return kwargs


class DFA(base.Automaton):
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
        kwargs = filter_kwargs(states, atoms, trans_dict, init_state, final)
        super(DFA, self).__init__(**kwargs,
                                  is_deterministic=True,
                                  acc_cond=(base.Automaton.ACC_REACH, 0))


class Monitor(base.Automaton):
    """
    Represents a Safety automaton.

    - Acceptance Type: `base.Automaton.ACC_SAFETY`
    - Acceptance condition: `(Safety, 0)`

        - **Accepts:** Infinite words.
        - **Interpretation:** :math:`\mathsf{Occ}(\\rho) \subseteq F` where :math:`F = \{q \in Q \mid \mathsf{AccSet}(q) = 0\}`

    - Number of Acceptance Sets: `1`
    - `final(state)` function returns either `-1` to indicate that the state is not accepting or `0` to
      indicate that the state is accepting with acceptance set `0`.
    """
    def __init__(self, states=None, atoms=None, trans_dict=None, init_state=None, final=None):
        """
        Constructs a Monitor.

        :param states: (Iterable) An iterable over states in the automaton.
        :param atoms: (Iterable[str]) An iterable over atomic propositions in the automaton.
        :param trans_dict: (dict) A dictionary defining the (deterministic) transition function of automaton.
                      Format of dictionary: {state: {logic.PLFormula: state}}
        :param init_state: (object) The initial state, a member of states iterable.
        :param final: (Iterable[states]) The set of final states, a subset of states iterable.
        """
        kwargs = filter_kwargs(states, atoms, trans_dict, init_state, final)
        super(Monitor, self).__init__(**kwargs,
                                      is_deterministic=True,
                                      acc_cond=(base.Automaton.ACC_SAFETY, 0))


class DBA(base.Automaton):
    """
    Represents a Deterministic Buchi automaton.

    - Acceptance Type: `base.Automaton.ACC_BUCHI`
    - Acceptance condition: `(Buchi, 0)`

        - **Accepts:** Infinite words.
        - **Interpretation:** :math:`\mathsf{Inf}(\\rho) \\cap F \\neq \\emptyset` where :math:`F = \{q \in Q \mid \mathsf{AccSet}(q) = 0\}`

    - Number of Acceptance Sets: `1`
    - `final(state)` function returns either `-1` to indicate that the state is not accepting or `0` to
      indicate that the state is accepting with acceptance set `0`.
    """
    def __init__(self, states=None, atoms=None, trans_dict=None, init_state=None, final=None):
        """
        Constructs a DBA.

        :param states: (Iterable) An iterable over states in the automaton.
        :param atoms: (Iterable[str]) An iterable over atomic propositions in the automaton.
        :param trans_dict: (dict) A dictionary defining the (deterministic) transition function of automaton.
                      Format of dictionary: {state: {logic.PLFormula: state}}
        :param init_state: (object) The initial state, a member of states iterable.
        :param final: (Iterable[states]) The set of final states, a subset of states iterable.
        """
        kwargs = filter_kwargs(states, atoms, trans_dict, init_state, final)
        super(DBA, self).__init__(**kwargs,
                                  is_deterministic=True,
                                  acc_cond=(base.Automaton.ACC_BUCHI, 0))


class DCBA(base.Automaton):
    """
    Represents a Deterministic co-Buchi automaton.

    - Acceptance Type: `base.Automaton.ACC_COBUCHI`
    - Acceptance condition: `(co-Buchi, 0)`

        - **Accepts:** Infinite words.
        - **Interpretation:** :math:`\mathsf{Inf}(\\rho) \\subseteq F` where :math:`F = \{q \in Q \mid \mathsf{AccSet}(q) = 0\}`

    - Number of Acceptance Sets: `1`
    - `final(state)` function returns either `-1` to indicate that the state is not accepting or `0` to
      indicate that the state is accepting with acceptance set `0`.
    """
    def __init__(self, states=None, atoms=None, trans_dict=None, init_state=None, final=None):
        """
        Constructs a DCBA.

        :param states: (Iterable) An iterable over states in the automaton.
        :param atoms: (Iterable[str]) An iterable over atomic propositions in the automaton.
        :param trans_dict: (dict) A dictionary defining the (deterministic) transition function of automaton.
                      Format of dictionary: {state: {logic.PLFormula: state}}
        :param init_state: (object) The initial state, a member of states iterable.
        :param final: (Iterable[states]) The set of final states, a subset of states iterable.
        """
        kwargs = filter_kwargs(states, atoms, trans_dict, init_state, final)
        super(DCBA, self).__init__(**kwargs,
                                   is_deterministic=True,
                                   acc_cond=(base.Automaton.ACC_COBUCHI, 0))


class DPA(base.Automaton):
    """
    Represents a Deterministic Buchi automaton.

    - Acceptance Type: `base.Automaton.ACC_PARITY`
    - Acceptance condition: `(Parity Min Even , 0)`

        - **Accepts:** Infinite words.
        - **Interpretation:** :math:`\\min_{q \\in \\mathsf{Inf}(\\rho)} \\chi(q)` is even, where :math:`\\chi: Q \\rightarrow \\mathbb{N}` is a coloring function that associates every state in DPA with a color (an integer).

    - Number of Acceptance Sets: `k`, where `k` is a positive integer.
    - `final(state)` a value between `0` and `k-1` to indicate that the color of a state.

    .. note:: The DPA definition is not stable. DO NOT USE IT!
    """
    def __init__(self, states=None, atoms=None, trans_dict=None, init_state=None, final=None):
        """
        Constructs a DPA.

        :param states: (Iterable) An iterable over states in the automaton.
        :param atoms: (Iterable[str]) An iterable over atomic propositions in the automaton.
        :param trans_dict: (dict) A dictionary defining the (deterministic) transition function of automaton.
                      Format of dictionary: {state: {logic.PLFormula: state}}
        :param init_state: (object) The initial state, a member of states iterable.
        :param final: (dict[state: int]) A state to color mapping.
        """
        kwargs = filter_kwargs(states, atoms, trans_dict, init_state, None)
        super(DPA, self).__init__(**kwargs,
                                  is_deterministic=True,
                                  acc_cond=(base.Automaton.ACC_PARITY, 0))
