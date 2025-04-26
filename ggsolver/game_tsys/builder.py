import pprint
import sys
import textwrap
import time
from abc import ABC, abstractmethod
from typing import Iterable

import networkx as nx
from loguru import logger
from tqdm import tqdm

from ggsolver.game_tsys.gamedef import BaseState


class Builder(ABC):
    """
    Abstract base class for building a solver.
    """

    def __init__(self, game_def: 'TSys', **kwargs):
        # Class variables
        self.game_def = game_def
        self.model = None
        self.build_report = "Model is not built yet."
        self.options = {
            "pointed": kwargs.get("pointed", True),
            "show_report": kwargs.get("show_report", True),
            "build_labels": kwargs.get("build_labels", False),
            "show_progress": kwargs.get("show_progress", True),
            "multicore": kwargs.get("multicore", False),
            "debug": kwargs.get("debug", False),
        }

        # Class configuration
        if game_def.is_deterministic:
            self._get_next_states = self._get_transitions_deterministic
        elif game_def.is_stochastic and game_def.is_qualitative:
            self._get_next_states = self._get_transitions_qual_probabilistic
        elif game_def.is_stochastic and not game_def.is_qualitative:
            self._get_next_states = self._get_transitions_probabilistic
        else:
            raise ValueError(
                "Could not determine transition type of transition system. "
                "The functions `is_deterministic`, `is_probabilistic`, `is_qualitative` all returned `False`."
            )

    # ----------------------------------------------------------------
    # Builder functionality
    # ----------------------------------------------------------------
    def build(self):
        # Start timer
        logger.info(f"Starting build with {self.options}.")
        start_time = time.time()

        # Pointed construction
        if self.options["pointed"]:
            init_states = self.get_init_states()
            assert isinstance(init_states, Iterable), \
                f"Invalid initial states: Expected an iterable of `gamedef.BaseState` instances, got {type(init_states)}."
            assert all(isinstance(s, BaseState) for s in init_states), \
                f"Invalid initial states: Expected an iterable of `gamedef.BaseState` instances, got {[type(s) for s in init_states]}."
            assert len(init_states) > 0, f"No initial states found for {self.game_def}."

            # Pointed multi-core construction
            if self.options["multicore"]:
                self._build_pointed_multi_core(init_states)

            # Pointed single-core construction
            else:
                self._build_pointed_single_core(init_states)
            # self.build_pointed(init_states)

        # Unpointed multi-core construction
        elif self.options["multicore"]:
            self._build_unpointed_multi_core()

        # Unpointed single-core construction
        else:
            self._build_unpointed_single_core()

        # Set graph-level attributes
        self.model.graph["name"] = self.game_def.name
        self.model.graph["model_type"] = self.game_def.model_type
        self.model.graph["is_qualitative"] = self.game_def.is_qualitative
        try:
            self.model.graph["state_vars"] = self.game_def.state_vars()
        except NotImplementedError:
            self.model.graph["state_vars"] = None

        # Build labels
        if self.options["build_labels"]:
            self.model.graph["atoms"] = self.game_def.atoms()
            if self.options["multicore"]:
                self._build_labels_multi_core()
            else:
                self._build_labels_single_core()

        # Stop timer
        end_time = time.time()

        # Generate report
        self.build_report = self.generate_build_report(run_time=end_time - start_time)
        if self.options["show_report"]:
            print(textwrap.dedent(textwrap.dedent(self.build_report)))

        return self.model

    def _build_pointed_single_core(self, init_states):
        # TODO: Frontier etc. should deal with state id's. Extract states wherever necessary.
        # Instantiate model
        self.instantiate_model()

        # Add initial states
        for state in init_states:
            assert isinstance(state, BaseState), \
                f"Invalid state type: Expected instance of `gamedef.BaseState` got {type(state)}."
            self.set_init_state(state)

        # FIFO exploration
        frontier = list(self.get_states(as_state_obj=True))
        frontier_set = set(frontier)
        with tqdm(total=len(frontier), disable=not self.options["show_progress"]) as pbar:
            while frontier:
                # Visit next state
                state = frontier.pop(0)
                frontier_set.remove(state)
                if self.options.get("debug", False):
                    logger.debug("State: \n" + pprint.pformat(state))

                # Get actions to explore from current state
                actions = self.get_actions(state)
                if len(actions) == 0:
                    logger.warning(f"No actions enabled at state:{state}. No transitions added.")
                    pbar.total = pbar.n + len(frontier)
                    pbar.update(1)
                    pbar.set_description(f"model.states: {self.get_num_states()}")
                    pbar.refresh()
                    continue

                if self.options.get("debug", False):
                    logger.debug("Actions: \n" + pprint.pformat(actions))
                    # logger.debug(pprint.pformat(f"Actions: {actions}"))

                # Apply all actions to generate next states
                transitions = self._get_next_states(state, actions)
                if not transitions:
                    raise ValueError(
                        f"Transition function returned empty set of transitions for state {state} and actions {actions}."
                    )
                for _, next_state, act_name, prob in transitions:
                    if self.options.get("debug", False):
                        logger.debug(f"Transition:\n\t{state=}\n\t{next_state=}\n\t{act_name=}\n\t{prob=}")
                    # Add next state, if not already in game graph
                    # if next_state not in (frontier_set | set(model.states(as_names=True))):
                    # if next_state not in set(self.get_states(as_state_obj=True)):  # TODO: Is this slowing down the loop?
                    if not self.has_state(next_state):
                        self.add_state(next_state)
                        if next_state not in frontier_set:
                            frontier.append(next_state)
                            frontier_set.add(next_state)
                    # Add transition (transition representation in GraphGame is (u, v, a))
                    self.add_transition(
                        from_state=state,
                        to_state=next_state,
                        action=act_name,
                        probability=prob
                    )

                # Update progress bar
                pbar.total = pbar.n + len(frontier)
                pbar.update(1)
                pbar.set_description(f"model.states: {self.get_num_states()}")
                pbar.refresh()

        # Update graph-level attributes
        try:
            self.set_state_vars(self.get_state_vars())
        except NotImplementedError:
            self.set_state_vars(None)

    def _build_pointed_multi_core(self, init_states):
        pass

    def _build_unpointed_single_core(self):
        pass

    def _build_unpointed_multi_core(self):
        pass

    def _build_labels_single_core(self):
        for node, data in self.model.nodes(data=True):
            state = data["state"]
            label = self.game_def.label(state)
            assert all(isinstance(p, str) for p in label), "Labels must all be strings."
            self.model.nodes[node]["label"] = label

    def _build_labels_multi_core(self):
        pass

    # ----------------------------------------------------------------
    # Helper functions
    # ----------------------------------------------------------------
    def _get_transitions_deterministic(self, state, actions):
        transitions = set()
        for action in actions:
            n_state = self.game_def.delta(state, action)
            assert isinstance(n_state, BaseState), \
                f"Invalid state type: Expected instance of `gamedef.BaseState` got {type(n_state)}."
            transitions.add((state, n_state, action, None))  # (action, next_state, prob)

        return transitions

    def _get_transitions_qual_probabilistic(self, state, actions):
        transitions = set()
        for action in actions:
            n_state_list = self.game_def.delta(state, action)
            assert isinstance(n_state_list, (set, tuple, list)), \
                (f"Qualitative probabilistic transition function must return "
                 f"an iterable of `tsys.State` or derived objects.")
            for n_state in n_state_list:
                assert isinstance(n_state, BaseState), \
                    f"Invalid state type: Expected instance of `tsys.State` got {type(n_state)}."
                transitions.add((state, n_state, action, None))  # (action, next_state, prob)
        return transitions

    def _get_transitions_probabilistic(self, state, actions):
        transitions = set()
        for action in actions:
            n_state_dict = self.game_def.delta(state, action)
            assert isinstance(n_state_dict, dict), f"Probabilistic transition function must return a dictionary."
            assert sum(n_state_dict.values()) - 1.0 < 1e-6, \
                f"Sum of probabilities must be 1. It is {sum(n_state_dict.values())}."
            for n_state, prob in n_state_dict.items():
                assert isinstance(n_state, BaseState), \
                    f"Invalid state type: Expected instance of `gamedef.BaseState` got {type(n_state)}."
                transitions.add((state, n_state, action, prob))  # (action, next_state, prob)

        return transitions

    # ----------------------------------------------------------------
    # Abstract functions (to be customized for model representation)
    # ----------------------------------------------------------------
    @abstractmethod
    def instantiate_model(self):
        pass

    @abstractmethod
    def add_state(self, state):
        pass

    @abstractmethod
    def add_transition(self, from_state, to_state, action, probability=None):
        pass

    @abstractmethod
    def has_state(self, state):
        pass

    @abstractmethod
    def get_states(self, as_state_obj=False, as_dict=False):
        pass

    @abstractmethod
    def get_init_states(self):
        pass

    @abstractmethod
    def get_state_vars(self):
        pass

    @abstractmethod
    def get_all_actions(self):
        pass

    @abstractmethod
    def get_label(self, state):
        pass

    @abstractmethod
    def set_init_state(self, state):
        pass

    @abstractmethod
    def get_actions(self, state):
        pass

    @abstractmethod
    def get_num_states(self):
        pass

    @abstractmethod
    def get_num_transitions(self):
        pass

    @abstractmethod
    def set_state_vars(self, var_names):
        pass

    @abstractmethod
    def set_label(self, state, label):
        pass

    # ----------------------------------------------------------------
    # Public functions
    # ----------------------------------------------------------------
    def generate_build_report(self, run_time):
        if self.model is not None:
            labels = "Not built"
            if self.options["build_labels"]:
                atom2state = {"/no-label": 0}
                for sid, state in self.get_states(as_dict=True).items():
                    label_sid = self.get_label(state)
                    if len(label_sid) == 0:
                        atom2state["/no-label"] += 1

                    for atom in label_sid:
                        atom2state[atom] = atom2state.get(atom, 0) + 1

                labels = "\n".join(
                    " " * 16 +
                    f"- {atom}: "
                    f"{atom2state[atom]} "
                    f"states"
                    for atom in atom2state.keys()
                )
                labels = "\n" + labels + "\n"

            state_vars = "not set"
            if isinstance(self.model.graph["state_vars"], (set, tuple, list)):
                state_vars = tuple(name for name in self.model.graph["state_vars"])

            output = \
                f"""
                ===============================
                Model Build Report
                ===============================
                Model type: {self.model.graph["model_type"]}
                Model name: {self.model.graph["name"]}
                State components: {state_vars}
                States: {self.get_num_states()}
                Actions: {len(self.get_all_actions())}
                Transitions: {self.get_num_transitions()}
                Initial states: {len(self.get_init_states())}
                Labels: {labels}
                Time taken: {run_time:.6f} seconds
                Memory used: {sys.getsizeof(self.model)} bytes
                ===============================

                """

            return output


class BuildGameGraph(Builder):
    def __init__(self, game_def: 'TSys', **kwargs):
        super().__init__(game_def, **kwargs)

        # Class variables
        self._state_to_sid_map = dict()

    def instantiate_model(self):
        self.model = nx.MultiDiGraph()

    def add_state(self, state):
        # Check if the state already exists in the map
        if state in self._state_to_sid_map:
            return self._state_to_sid_map[state]

        # Assign a new state ID and add the state to the graph
        state_id = len(self.model.nodes)
        self.model.add_node(state_id, state=state)

        # Update the state-to-ID map
        self._state_to_sid_map[state] = state_id

        return state_id

    def add_transition(self, from_state, to_state, action, probability=None):
        uid = self.add_state(from_state)
        vid = self.add_state(to_state)
        self.model.add_edge(uid, vid, action=action, probability=probability)
        self.model.graph["actions"] = self.model.graph.get("actions", set()) | {action}

    def has_state(self, state):
        return state in self._state_to_sid_map

    def get_states(self, as_state_obj=False, as_dict=False):
        if as_state_obj:
            return [data["state"] for _, data in self.model.nodes(data=True)]
        if as_dict:
            return {node: data["state"] for node, data in self.model.nodes(data=True)}
        return list(self.model.nodes())

    def get_init_states(self):
        try:
            return self.game_def.init_states()
        except NotImplementedError:
            return self.game_def.states()

    def get_state_vars(self):
        return self.game_def.state_vars()

    def get_num_states(self):
        return self.model.number_of_nodes()

    def get_actions(self, state):
        return self.game_def.actions(state)

    def get_all_actions(self):
        return self.model.graph["actions"]

    def get_label(self, state):
        state_id = self.add_state(state)
        return self.model.nodes[state_id]["label"]

    def get_num_transitions(self):
        return len(self.model.edges())

    def set_init_state(self, state):
        state_id = self.add_state(state)
        self.model.graph["init_states"] = self.model.graph.get("init_states", set()) | {state_id}

    def set_state_vars(self, var_names):
        self.model.graph["state_vars"] = var_names

    def set_label(self, state, label):
        state_id = self.add_state(state)
        self.model.nodes[state_id]["label"] = label


class BuildMatrixGame(Builder):
    pass


class BuildDictGame(Builder):
    pass


class BuildBDDGame(Builder):
    pass
