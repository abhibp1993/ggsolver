import itertools
from abc import ABC, abstractmethod
from collections import namedtuple
from collections.abc import Callable
from operator import itemgetter
from typing import List, Dict, Union, Any, Iterable, Optional

import ggsolver.game_tsys.utils as utils
from ggsolver.game_tsys.constants import ModelType


class BaseState:
    pass


def new_state(class_name: str, components: List[str]):
    fields = tuple(components)

    def __new__(cls, *args, **kwargs):
        if args and kwargs:
            raise TypeError("Cannot mix positional and keyword arguments")
        if kwargs:
            if set(kwargs.keys()) != set(fields):
                missing = set(fields) - set(kwargs.keys())
                extra = set(kwargs.keys()) - set(fields)
                parts = []
                if missing:
                    parts.append(f"missing fields: {', '.join(missing)}")
                if extra:
                    parts.append(f"unknown fields: {', '.join(extra)}")
                raise TypeError("; ".join(parts))
            args = tuple(kwargs[name] for name in fields)
        elif len(args) != len(fields):
            raise TypeError(f"Expected {len(fields)} arguments, got {len(args)}:{args}.")
        return tuple.__new__(cls, args)

    def __repr__(self):
        values = ', '.join(f"{name}={getattr(self, name)}" for name in fields)
        return f"{class_name}({values})"

    def __eq__(self, other: list | tuple | BaseState | dict):
        if isinstance(other, dict):
            return self.as_dict() == other
        else:
            return tuple.__eq__(self, tuple(other))

    def __iter__(self):
        return tuple.__iter__(self)

    def __reduce__(self):
        return (cls, tuple(self))

    def __hash__(self):
        return hash(tuple(self))

    def as_dict(self):
        return {name: getattr(self, name) for name in fields}

    def replace(self, **kwargs):
        current = self.as_dict()
        for k in kwargs:
            if k not in fields:
                raise TypeError(f"Unknown field: {k}")
        current.update(kwargs)
        return cls(**current)

    namespace = {
        '__slots__': (),
        '__new__': __new__,
        '__repr__': __repr__,
        '__eq__': __eq__,
        '__iter__': __iter__,
        '__hash__': __hash__,
        '__reduce__': __reduce__,
        'as_dict': as_dict,
        'replace': replace,
        'components': fields
    }

    for i, name in enumerate(fields):
        namespace[name] = property(itemgetter(i), doc='Property getter for ' + name)

    cls = type(class_name, (tuple, BaseState), namespace)
    cls.__module__ = '__main__'

    return cls


# ===================================================================
# ABSTRACT CLASS: TRANSITION SYSTEM
# ===================================================================
class TSys(ABC):
    def __init__(self, name: str, model_type: ModelType, *, is_qualitative=False, **kwargs):
        self._name = name
        self._model_type = model_type
        self._is_qualitative = is_qualitative

    def __repr__(self):
        return f"{self.__class__.__name__}(name={self.name}, model_type={self.model_type})"

    # ----------------------------------------------------------------
    # Properties
    # ----------------------------------------------------------------
    @property
    def name(self):
        return self._name

    @property
    def model_type(self):
        return self._model_type

    @property
    def is_qualitative(self):
        return self._is_qualitative

    @property
    def is_deterministic(self):
        return utils.is_deterministic(self.model_type)

    @property
    def is_stochastic(self):
        return utils.is_stochastic(self.model_type)

    # ----------------------------------------------------------------
    # Abstract methods
    # ----------------------------------------------------------------
    @abstractmethod
    def states(self):
        raise NotImplementedError

    @abstractmethod
    def actions(self, state):
        raise NotImplementedError

    @abstractmethod
    def delta(self, state, action):
        raise NotImplementedError

    # ----------------------------------------------------------------
    # Optional methods
    # ----------------------------------------------------------------
    def state_vars(self):
        raise NotImplementedError

    def init_states(self):
        raise NotImplementedError

    def atoms(self):
        raise NotImplementedError

    def label(self, state):
        raise NotImplementedError

    def reward(self, state, action=None):
        raise NotImplementedError

    # ----------------------------------------------------------------
    # Implemented methods
    # ----------------------------------------------------------------
    def turn(self, state: namedtuple):
        return state.turn

    # def build(self, **options):
    #     builder = Builder(self, **options)
    #     return builder.build()


# ===================================================================
# MODULE (A Prism-style implementation of transition system)
# ===================================================================
class Module(TSys):
    def __init__(self, name: str, model_type: ModelType, *, is_qualitative=False, **kwargs):
        super().__init__(name, model_type, is_qualitative=is_qualitative, **kwargs)

        # Specialized data structures for defining module
        self._state_vars = dict()
        self._state_validators = dict()
        self._init_states = set()
        self._actions = dict()
        self._enabled_actions = None
        self._atoms = dict()
        self._reward = dict()

        # State object
        self._state_class = None

    # ----------------------------------------------------------------
    # Public functions for defining module
    # ----------------------------------------------------------------
    def register_state_var(
            self,
            name: str,
            domain: Iterable
    ) -> None:
        assert name not in self._state_vars, f"State variable {name} already exists."
        self._state_vars[name] = set(domain)

    def register_state_vars(
            self,
            **vars_with_domains: Iterable[Any]
    ) -> None:
        for name, domain in vars_with_domains.items():
            self.register_state_var(name, domain)

    def register_state_validator(
            self,
            name: str,
            func: Callable[[BaseState], bool]
    ) -> None:
        assert name not in self._state_validators, f"State validator {name} already exists."
        self._state_validators[name] = func

    def register_state_validators(
            self,
            **validators: Callable[[BaseState], bool]
    ) -> None:
        for name, func in validators.items():
            self.register_state_validator(name, func)

    def add_init_state(
            self,
            state: BaseState
    ) -> None:
        assert isinstance(state, BaseState), "Input parameter `state` must be an instance of BaseState."
        self._init_states.add(state)

    def add_init_states(
            self,
            states: Iterable[BaseState]
    ) -> None:
        for state in states:
            self._init_states.add(state)

    def set_enabled_actions_func(
            self,
            func: Callable[[BaseState], Iterable[str]]
    ) -> None:
        self._enabled_actions = func

    def register_action(
            self,
            name: str,
            func: Callable[[BaseState], Union[BaseState, Iterable[BaseState], Dict[BaseState, float]]]
    ) -> None:
        assert name not in self._actions, f"Action {name} already exists."
        self._actions[name] = func

    def register_actions(
            self,
            **actions: Dict[str, Callable[[BaseState], Union[BaseState, Iterable[BaseState], Dict[BaseState, float]]]]
    ) -> None:
        for name, func in actions.items():
            self.register_action(name, func)

    def register_atom(
            self,
            name: str,
            func: Callable[[BaseState], bool]
    ) -> None:
        assert name not in self._atoms, f"Label {name} already exists."
        self._atoms[name] = func

    def register_atoms(
            self,
            **atoms: Callable[[BaseState], bool]
    ) -> None:
        for name, func in atoms.items():
            self.register_atom(name, func)

    def register_reward_func(
            self,
            name: str,
            func: Callable[[BaseState, Optional[str]], float]
    ):
        assert name not in self._reward, f"Reward function {name} already exists."
        self._reward[name] = func

    # ----------------------------------------------------------------
    # Transition system implementation
    # ----------------------------------------------------------------
    def states(self):
        # If state class is not defined, create it
        if self._state_class is None:
            self._state_class = new_state(self.name, list(self._state_vars.keys()))

        # Ensure state variables are defined
        if not self._state_vars:
            raise ValueError("State variables are not defined.")

        # Extract variable names and their domains
        var_names = list(self._state_vars.keys())
        domains = list(self._state_vars.values())

        # Generate Cartesian product of all domains
        all_states = (
            self._state_class(**dict(zip(var_names, values)))
            for values in itertools.product(*domains)
        )

        return all_states

    def actions(self, state):
        if self._enabled_actions:
            # Use the user-defined function to determine enabled actions
            return self._enabled_actions(state)
        else:
            # If no function is defined, assume all actions are available
            return list(self._actions.keys())

    def delta(self, state, action):
        # Check if the action is valid & enabled
        if action not in self._actions:
            raise ValueError(f"Action {action} is not defined.")
        if self._enabled_actions and action not in self._enabled_actions(state):
            raise ValueError(f"Action {action} is not enabled in the current state {state}.")

        # Apply action
        next_state = self._actions[action](state)

        # Validate state
        if self.is_deterministic:
            assert all(validator(next_state) for validator in self._state_validators.values()), \
                f"Next state {next_state} does not satisfy all state validators."
            return next_state

        else:  # self.is_probabilistic:
            assert all(
                validator(state)
                for validator in self._state_validators.values()
                for state in next_state
            ), \
                f"All next states in {next_state} must be BaseState or its derived class instances."
            assert all(
                validator(state)
                for validator in self._state_validators.values()
                for state in next_state
            ), \
                f"Next state {next_state} does not satisfy all state validators."
            if not self.is_qualitative:
                assert all(isinstance(v, (int, float)) for v in next_state.values()), \
                    f"Next state {next_state} must be a dictionary with float values."

            return next_state

    def state_vars(self):
        return self._state_vars.keys()

    def init_states(self):
        return self._init_states

    def atoms(self):
        return self._atoms.keys()

    def label(self, state):
        true_atoms = {atom for atom, func in self._atoms.items() if func(state)}
        return true_atoms

    def reward(self, state, action=None):
        if action is None:
            return {name: func(state) for name, func in self._reward.items()}
        else:
            return {name: func(state, action) for name, func in self._reward.items()}


if __name__ == '__main__':
    # Create a new state class
    State = new_state("State", ["x", "y", "z"])

    # Create instances of the State class
    state1 = State(1, 2, 3)
    state2 = State(1, 2, 3)
    state3 = State(4, 5, 6)

    # Test __repr__
    print("Testing __repr__:")
    print(state1)  # Expected: State(x=1, y=2, z=3)

    # Test __eq__
    print("\nTesting __eq__:")
    print(state1 == state2)  # Expected: True
    print(state1 == state3)  # Expected: False

    # Test __iter__
    print("\nTesting __iter__:")
    print(list(state1))  # Expected: [1, 2, 3]

    # Test attribute access
    print("\nTesting attribute access:")
    print(state1.x, state1.y, state1.z)  # Expected: 1 2 3

    # Test as_dict method
    print("\nTesting as_dict method:")
    print(state1.as_dict())  # Expected: {'x': 1, 'y': 2, 'z': 3}

    # Test replace method
    print("\nTesting replace method:")
    state4 = state1.replace(x=10, z=30)
    print(state4)  # Expected: State(x=10, y=2, z=30)

    # Test invalid initialization
    print("\nTesting invalid initialization:")
    try:
        invalid_state = State(1, 2)  # Missing one argument
    except TypeError as e:
        print(e)  # Expected: TypeError with a message about the number of arguments

    try:
        invalid_state = State(1, 2, 3, 4)  # Extra argument
    except TypeError as e:
        print(e)  # Expected: TypeError with a message about the number of arguments

    try:
        invalid_state = State(x=1, y=2, a=3)  # Invalid field name
    except TypeError as e:
        print(e)  # Expected: TypeError with a message about unknown fields
