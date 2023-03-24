Models
======


Automaton
---------

.. autoclass:: ggsolver.logic.base.Automaton
    :members:
    :inherited-members:
    :special-members: __init__


TSys
----

.. autoclass:: ggsolver.models.TSys
    :members:   __init__,
                states,
                actions,
                delta,
                atoms,
                label,
                initialize,
                graphify,
                graphify_pointed,
                graphify_unpointed,
                serialize,
                save,
                deserialize,
                load,
                init_state,
                is_deterministic,
                is_nondeterministic,
                is_probabilistic


Game
----

.. autoclass:: ggsolver.models.Game
    :members:   __init__,
                states,
                actions,
                delta,
                atoms,
                label,
                initialize,
                graphify,
                graphify_pointed,
                graphify_unpointed,
                serialize,
                save,
                deserialize,
                load,
                init_state,
                is_deterministic,
                is_nondeterministic,
                is_probabilistic,
                final,
                is_turn_based,
                turn,
                p1_acts,
                p2_acts,
                win_cond,
                formula


Solver
------

.. autoclass:: ggsolver.models.Solver
    :members:
    :inherited-members:


DeterministicStrategy
---------------------

.. autoclass:: ggsolver.models.DeterministicStrategy
    :members:
    :inherited-members:
    :special-members: __call__


NonDeterministicStrategy
------------------------

.. autoclass:: ggsolver.models.NonDeterministicStrategy
    :members:
    :inherited-members:
    :special-members: __call__
