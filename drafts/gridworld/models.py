"""
TODO. State machine: Connects ggsolver game graphs with pygame simulation.
    [+] State-based interface. That is, all user functions input and output states, actions
        even though the state machine is constructed using a game graph (Graph object).
    [+] Expose methods to interact with state machine: step_forward, step_backward, ...
    [+] Complete step_forward function (see gridworld package impl).
    [+] Limit history.
    * Make serializable to save and replay.

TODO. Reorganize event system with three ideas:
    [+] Events are pygame.Event objects.
    [+] Each Window listens to a subset of events from WINDOW_EVENTS.
    [+] Users can register new handlers for any event from WINDOW_EVENTS.
    [-] Each control has events and handlers variable.
        events: {event_name: event_condition}
        handler: {event_name: list_of_handlers}
    * Users can register new handlers for any event from Control.AVAILABLE_EVENTS.
    * Each control listens to a subset of events from Control.AVAILABLE_EVENTS.
    * All Controls have default handlers that are called first, then the user handlers.
    * An event is triggered for a Control if and only if it is in the "scope" of that Control object.
        E.g. mouse_click occurs if it occurs over that control.
    * Event handler receives `event_args` dictionary with relevant information.
    * `event_args` contains `sender: <sending object>` entry.
    * For keyboard events, `event_args` contains `key_code`s and `modifier` keys as str, not as pygame objects.
    * Double click event.
    * Default resize event.


TODO. General features
    [+] All positions are pygame.math.Vector2
    [+] All objects are rendered with respect to their parent.
    [+] Anchors for positioning Controls within Parent.
    [+] Control has `move, move_by` methods.
    [+] Control's parent can be changed using property setter
    * Controls are "Hoverable", "Selectable", "Clickable", "Draggable", "Hidden/Visible"
    * Connection to GUI controls.
    * Grid has `move_north, move_south, ...` methods.
    * Pop-up images (non-blocking).
    * Pop-up messages (blocking or non-blocking).
    * Pop-up input box (blocking or non-blocking).
    * Thought-bubble control (attached to Character).
    * Animated Characters.
    * Battery animated control.
        - Shows number of remaining levels.
        - Shows bars (if max battery level is exactly what is available in sprite sheet).

"""
import inspect
import sys

import pygame
import random

from ggsolver import util
from multiprocessing import Process
from scipy.stats import rv_discrete
from typing import List


# ===========================================================================================
# GLOBALS
# ===========================================================================================

COLOR_TRANSPARENT = pygame.Color(0, 0, 0, 0)   # The last 0 indicates 0 alpha, a transparent color


# ===========================================================================================
# EVENTS
#   Define custom events called GWSIM_EVENTS.
#   Each GWSIM_EVENT has a unique id, name, and the name of sender who triggered this event.
#
#   The following E_XYZ event names provide the unique id and name of event.
#   The sender and other arguments are determined while posting the event.
#   See the process_custom_events() functions in Window, Control classes.
# ===========================================================================================

E_SM_UPDATE = pygame.USEREVENT + 1
GWSIM_EVENTS = pygame.USEREVENT + 2
GWSIM_EVENTS_ID = {
    # "E_QUIT": 0,
    "E_KEYPRESS": 1,
    "E_KEYDOWN": 2,
    "E_KEYUP": 3,
    "E_RESIZE": 4,
    # "E_MOUSEMOTION": 5,
    "E_MOUSEBUTTONDOWN": 6,
    "E_MOUSEBUTTONUP": 7,
    "E_MOUSEBUTTONPRESS": 8,
    # "E_MOUSEWHEEL": 9,
    # "E_WINDOWMINIMIZED": 10,
    # "E_WINDOWMAXIMIZED": 11,
    # "E_WINDOWFOCUSGAINED": 12,
    # "E_WINDOWFOCUSLOST": 13,
    # "E_WINDOWMOVED": 14,
    # "E_WINDOWRESIZED": 15,
    # "E_WINDOWENTER": 16,
    # "E_WINDOWLEAVE": 17,
    # "E_WINDOWCLOSE": 18,
    "E_CELLCHANGED": 20,
    "E_MOUSECLICK": 21,
    "E_MOUSEDOUBLECLICK": 22,
    "E_MOUSEHOVER": 23,
    "E_WINDOWRESIZABILITYCHANGED": 24,
}

# E_NAME_QUIT = "PygameEvent(Game Quit)"
# E_NAME_KEYPRESS = "PygameEvent(Key Pressed)"
# E_NAME_KEYDOWN = "PygameEvent(Key Down)"
# E_NAME_KEYUP = "PygameEvent(Key Up)"
# E_NAME_RESIZE = "PygameEvent(Resize)"
# E_NAME_MOUSEMOTION = "PygameEvent(Mouse Moved)"
# E_NAME_MOUSEBUTTONDOWN = "PygameEvent(Mouse Button Down)"
# E_NAME_MOUSEBUTTONUP = "PygameEvent(Mouse Button Up)"
# E_NAME_MOUSEBUTTONPRESS = "PygameEvent(Mouse Button Pressed)"
# E_NAME_MOUSEWHEEL = "PygameEvent(Mouse Wheel)"
# E_NAME_WINDOWMINIMIZED = "PygameEvent(Window Minimized)"
# E_NAME_WINDOWMAXIMIZED = "PygameEvent(Window Maximized)"
# E_NAME_WINDOWFOCUSGAINED = "PygameEvent(Window Focus Gained)"
# E_NAME_WINDOWFOCUSLOST = "PygameEvent(Window Focus Lost)"
# E_NAME_WINDOWMOVED = "PygameEvent(Window Moved)"
# E_NAME_WINDOWRESIZED = "PygameEvent(Window Resized)"
# E_NAME_WINDOWENTER = "PygameEvent(Mouse Entered Window)"
# E_NAME_WINDOWLEAVE = "PygameEvent(Mouse Left Window)"
# E_NAME_WINDOWCLOSE = "PygameEvent(Window Closed)"

# E_NAME_SMUPDATE = "GWSimEvent(StateMachine Update)"
# E_NAME_CELLCHANGED = "GWSimEvent(Cell Changed)"
# E_NAME_MOUSECLICK = "GWSimEvent(Mouse Click)"
# E_NAME_MOUSEDOUBLECLICK = "GWSimEvent(Mouse Double Click)"
# E_NAME_MOUSEHOVER = "GWSimEvent(Mouse Hover)"
# E_NAME_WINDOWRESIZABILITYCHANGED = "GWSimEvent(Window Resizability Changed)"
#
# E_CELLCHANGED = pygame.event.Event(GWSIM_EVENTS, name=E_NAME_CELLCHANGED)
# E_MOUSECLICK = pygame.event.Event(GWSIM_EVENTS, name=E_NAME_MOUSECLICK)
# E_MOUSEDOUBLECLICK = pygame.event.Event(GWSIM_EVENTS, name=E_NAME_MOUSEDOUBLECLICK)
# E_MOUSEHOVER = pygame.event.Event(GWSIM_EVENTS, name=E_NAME_MOUSEHOVER)
# E_SMUPDATE = pygame.event.Event(GWSIM_EVENTS, name=E_NAME_SMUPDATE)
# E_WINDOWRESIZABILITYCHANGED = pygame.event.Event(GWSIM_EVENTS, name=E_NAME_WINDOWRESIZABILITYCHANGED)

#
# CONTROL_EVENTS = [
#     E_NAME_KEYPRESS,
#     E_NAME_KEYDOWN,
#     E_NAME_KEYUP,
#     E_NAME_RESIZE,
#     E_NAME_MOUSEMOTION,
#     E_NAME_MOUSEBUTTONDOWN,
#     E_NAME_MOUSEBUTTONUP,
#     E_NAME_MOUSEBUTTONPRESS,
#     E_NAME_MOUSEWHEEL,
# ]
#
# WINDOW_EVENTS = [
#     E_NAME_QUIT,
#     E_NAME_WINDOWMINIMIZED,
#     E_NAME_WINDOWMAXIMIZED,
#     E_NAME_WINDOWFOCUSLOST,
#     E_NAME_WINDOWFOCUSGAINED,
#     E_NAME_WINDOWMOVED,
#     E_NAME_WINDOWRESIZED,
#     E_NAME_WINDOWENTER,
#     E_NAME_WINDOWLEAVE,
#     E_NAME_WINDOWCLOSE,
#     E_NAME_WINDOWRESIZABILITYCHANGED,
#     E_NAME_SMUPDATE,
#     E_NAME_KEYDOWN,
#     E_NAME_KEYUP,
# ]


# ===========================================================================================
# ENUMERATIONS
# ===========================================================================================
class BorderStyle:
    SOLID = "solid"
    HIDDEN = "hidden"


class GridLayout:
    AUTO = "auto"
    CUSTOM = "custom"


class GameMode:
    AUTO = "auto"
    MANUAL = "manual"


class AnchorStyle:
    NONE = "None"
    TOP_LEFT = "Top-left"
    CENTER = "Center"


# ===========================================================================================
# SIMULATION OBJECTS
# ===========================================================================================
class Window2:
    ACTIVE_WINDOWS: List['Window'] = []

    def __init__(self, name, size, **kwargs):
        """
        :param name: (str) Name of window
        :param size: (tuple[int, int]) Size of window

        kwargs:
        * title: (str) Window title (Default: "Window")
        * resizable: (bool) Can the window be resized? (Default: False)
        * fps: (float) Frames per second for pygame simulation. (Default: 60)
        * conn: (multiprocessing.Connection) Connection to another window.
        * backcolor: (tuple[int, int, int]) Default backcolor of window. (Default: (0, 0, 0))
        """
        # # Initialize pygame
        # pygame.init()

        # Instance variables
        self._name = name
        self._controls = dict()
        self._sprites = pygame.sprite.LayeredUpdates()
        self._title = f"Window"
        self._size = size
        self._backcolor = kwargs["backcolor"] if "backcolor" in kwargs else (0, 0, 0)
        self._resizable = kwargs["resizable"] if "resizable" in kwargs else False
        self._fps = kwargs["fps"] if "fps" in kwargs else 60
        self._running = False

        # # Initialize pygame window
        # pygame.display.set_caption(self._title)
        # try:
        #     pygame.display.set_icon(pygame.image.load("sprites/GWSim.png"))
        # except FileNotFoundError:
        #     pass
        # self._screen = pygame.display.set_mode([self.width, self.height], pygame.RESIZABLE)

        # Add current window to active windows
        Window.ACTIVE_WINDOWS.append(self)

    def __del__(self):
        Window.ACTIVE_WINDOWS.remove(self)

    # def __getstate__(self):
    #     state = self.__dict__.copy()
    #     # surface = state.pop("_screen")
    #     # state["_screen"] = (pygame.image.tostring(surface, "RGB"), surface.get_size())
    #     return state

    def delta(self):
        pass

    def update(self, screen):
        # print(f"Called: {self}.{inspect.stack()[0][3]}")
        # Clear previous drawing
        screen.fill(self._backcolor)

        # Update all controls (sprites)
        # self._sprites.update()
        # self._sprites.draw(self._screen)

        # Update screen
        pygame.display.flip()

    def handle_event(self, event):
        # Handle special pygame-level events
        if event.type == pygame.QUIT:
            self.on_exit(None)

        if event.type == pygame.WINDOWRESIZED:
            # FIXME: Decide the arguments for on_resize
            #  (WINDOWMOVED, WINDOWRESIZED and WINDOWSIZECHANGED have x and y attributes)
            self.on_resize(None)

        if event.type == pygame.WINDOWMINIMIZED:
            self.on_minimize(None)

        if event.type == pygame.WINDOWMAXIMIZED:
            self.on_maximize(None)

        if event.type == pygame.WINDOWENTER:
            self.on_mouse_enter(None)

        if event.type == pygame.WINDOWLEAVE:
            self.on_mouse_leave(None)

        if event.type == pygame.WINDOWFOCUSGAINED:
            self.on_focus_gained(None)

        if event.type == pygame.WINDOWFOCUSLOST:
            self.on_focus_lost(None)

        # # Message events
        # # TODO. Use multiprocessing Queue to get and send.
        # #  Do we need sender and receiver as two queues? Or just one shared queue suffices?
        # # If message is received, raise on_msg_received event.

        # Pass the event to child controls
        for name, control in self._controls.items():
            control.process_event(event)

    def run(self, args=None):
        # Initialize pygame
        pygame.init()

        pygame.display.set_caption(self._title)
        try:
            pygame.display.set_icon(pygame.image.load("sprites/GWSim.png"))
        except FileNotFoundError:
            pass
        screen = pygame.display.set_mode([self.width, self.height], pygame.RESIZABLE)

        clock = pygame.time.Clock()
        self._running = True
        while self._running:
            # Handle handle_event
            for event in pygame.event.get():
                self.handle_event(event)

            # Update window state and sprite visualization update.
            self.delta()
            self.update(screen)

            # Set FPS
            clock.tick(self._fps)

    def stop(self):
        self._running = False

    def add_control(self, control):
        self._controls[control.name] = control
        self._sprites.add(control)

    def rem_control(self, control):
        # Remove control, if exists, from controls list.
        if isinstance(control, str):
            control = self._controls.pop(control, None)
        else:
            control = self._controls.pop(control.name, None)

        # Remove the control from sprite group, if exists.
        if control is not None:
            self._sprites.remove(control)

    def create_control(self, cls_control, constructor_kwargs):
        # Preprocess input arguments (basic control arguments, any addtional parameters should be passed by user)
        assert "name" in constructor_kwargs, "constructor_kwargs must have 'name' parameter."
        assert "size" in constructor_kwargs, "constructor_kwargs must have 'size' parameter."
        constructor_kwargs["parent"] = self
        constructor_kwargs["position"] = constructor_kwargs["position"] if "position" in constructor_kwargs else (0, 0)

        # Construct control
        control = cls_control(**constructor_kwargs)

        # Add control to window
        self.add_control(control)

    # ===========================================================================
    # PROPERTIES
    # ===========================================================================
    @property
    def screen(self):
        return self._screen

    @property
    def image(self):
        return self._screen

    @property
    def rect(self):
        return self._screen.get_rect()

    @property
    def controls(self):
        return self._controls

    @property
    def name(self):
        return self._name

    @property
    def title(self):
        return self._title

    @title.setter
    def title(self, value):
        self._title = value

    @property
    def width(self):
        return self._size[0]

    @width.setter
    def width(self, value):
        raise NotImplementedError("TODO. Raise resize() event.")

    @property
    def height(self):
        return self._size[1]

    @height.setter
    def height(self, value):
        raise NotImplementedError("TODO. Raise resize() event.")

    @property
    def resizable(self):
        return self._resizable

    @resizable.setter
    def resizable(self, value):
        raise NotImplementedError("TODO. Raise resize() event.")

    @property
    def backcolor(self):
        return self._backcolor

    @backcolor.setter
    def backcolor(self, value):
        self._backcolor = value

    @property
    def fps(self):
        return self._fps

    @fps.setter
    def fps(self, value):
        self._fps = value

    def get_mouse_position(self):
        return pygame.mouse.get_pos()

    # ===========================================================================
    # EVENTS
    # ===========================================================================
    def on_msg_received(self, event_args):
        pass
        # print(f"Called: {inspect.stack()[0][3]}")

    def on_exit(self, event_args):
        # for name, control in self._controls.items():
        #     control.on_exit(event_args)
        self._running = False

    def on_resize(self, event_args):
        pass
        # print(f"Called: {inspect.stack()[0][3]}")

    def on_minimize(self, event_args):
        pass
        # print(f"Called: {inspect.stack()[0][3]}")

    def on_maximize(self, event_args):
        pass
        # print(f"Called: {inspect.stack()[0][3]}")

    def on_mouse_enter(self, event_args):
        pass
        # print(f"Called: {inspect.stack()[0][3]}")

    def on_mouse_leave(self, event_args):
        pass
        # print(f"Called: {inspect.stack()[0][3]}")

    def on_focus_gained(self, event_args):
        pass
        # print(f"Called: {inspect.stack()[0][3]}")

    def on_focus_lost(self, event_args):
        pass
        # print(f"Called: {inspect.stack()[0][3]}")


class StateMachine:
    def __init__(self, graph):
        """
        Programmer's Notes:
            * Assume a graph property "actions" is available since game is gridworld.
        """
        # State machine
        self._graph = graph
        self._curr_time_step = 0
        self._state_history = []
        self._action_history = []
        self._memory_limit = float("inf")

        # Cache
        self._state_to_node = dict()
        self._actions = self._graph["actions"]
        self._cache_state_to_node()
        
    def initialize(self, state):
        node = self.state_to_node(state)
        if node in self._graph.nodes():
            self.reset()
            self._state_history.append(state)

    def reset(self):
        pass

    def step_forward(self, act, choice_function=None, *args, **kwargs):
        """
        One-step forward.

        :param choice_function: (function) A function that inputs list of states and returns a single state.

        kwargs:
            * `override_act`: (bool) When len(state hist) > curr_time_step AND override_act is True,
                the previously witnessed future is cleared and the game restarts at current point. (Default: False)
        """
        # Was the game stepped backward?
        if len(self._state_history) - 1 > self._curr_time_step:
            override_act = kwargs["override_act"] if "override_act" in kwargs else False
            if override_act:
                self._state_history = self._state_history[0: self._curr_time_step + 1]
                self._action_history = self._action_history[0: self._curr_time_step + 1]
            else:
                self._curr_time_step += 1
                return

        # Validate action
        if act not in self._actions:
            raise ValueError(f"SM.step_forward called with invalid action: {act}. Acceptable: {self._actions}.")

        # If choice function is not provided by user, use default
        choice_function = choice_function if choice_function is not None else self._default_choice_function

        # Get current node and its out_edges
        curr_node = self.state_to_node(self.curr_state)
        out_edges = self._graph.out_edges(curr_node)

        # Determine next state
        next_state = None
        if self._graph["is_deterministic"]:
            for uid, vid, key in out_edges:
                if self._graph["input"][uid, vid, key] == act:
                    next_state = self.node_to_state(vid)
                    break

        else:  # either non-deterministic or probabilistic
            successors = []
            for uid, vid, key in out_edges:
                if self._graph["input"][uid, vid, key] == act:
                    successors.append(self.node_to_state(vid))

            next_state = choice_function(successors, *args, **kwargs)

        # Update current state, histories and time
        self._state_history.append(next_state)
        self._action_history.append(act)
        if len(self._state_history) > self._memory_limit:
            self._state_history.pop(0)
            self._action_history.pop(0)
        else:
            self._curr_time_step += 1

    def step_forward_n(self, actions, n):
        """
        Step forward `n`-steps.
        Note: `n` is a param because actions are often list/tuple, which could lead to confusion.
        """
        assert len(actions) == n
        for act in actions:
            self.step_forward(act)

    def step_backward(self, n=1, clear_history=False):
        """
        Step backward by `n` steps.
        When clear_history is True, the last `n` steps are cleared.

        Note: initial state cannot be cleared. It must be reinitialized using initialize() function.
        """
        if self._curr_time_step - n < 0:
            # TODO. Show a warning message.
            return

        if clear_history:
            self._state_history = self._state_history[:len(self._state_history) - n]
            self._action_history = self._action_history[:len(self._action_history) - n]
            self._curr_time_step -= n

        else:
            self._curr_time_step -= n

    def state_to_node(self, state):
        return self._state_to_node[state]

    def node_to_state(self, node):
        return self._graph["state"][node]

    def _cache_state_to_node(self):
        np_state = self._graph["state"]
        for node in self._graph.nodes():
            self._state_to_node[np_state[node]] = node

    def _default_choice_function(self, choices, *args, **kwargs):
        if isinstance(choices, rv_discrete):
            # TODO. Need to figure out how to use scipy rv_discrete.
            raise NotImplementedError("Need to figure out how to use scipy rv_discrete.")
        else:
            return random.choice(choices)

    def states(self):
        return (self.node_to_state(node) for node in self._graph.nodes())

    def actions(self):
        return self._actions

    def delta(self, state, act):
        """
        Returns a list of next states possible on applying the action at given state.

        Programmer's Note:
            * This function is only used for inspection purposes. It does not affect "step" functions.
        """
        # Get current node and its out_edges
        curr_node = self.state_to_node(self.curr_state)
        out_edges = self._graph.out_edges(curr_node)

        successors = []
        for uid, vid, key in out_edges:
            if self._graph["input"][uid, vid, key] == act:
                successors.append(self.node_to_state(vid))

        return successors

    def get_node_property(self, p_name, state):
        return self._graph[p_name][self.state_to_node(state)]

    def get_edge_property(self, p_name, from_state, act, to_state):
        from_node = self.state_to_node(from_state)
        to_node = self.state_to_node(to_state)
        for uid, vid, key in self._graph.out_edges(from_node):
            if vid == to_node and self._graph["input"][uid, vid, key] == act:
                return self._graph[p_name][uid, vid, key]
        raise ValueError(f"Edge property:{p_name} is undefined for transition (u:{from_state}, v:{to_state}, a:{act})")

    def get_graph_property(self, p_name):
        return self._graph[p_name]

    @property
    def curr_state(self):
        return self._state_history[self._curr_time_step]

    @curr_state.setter
    def curr_state(self, state):
        """ Sets the current state to given state. """
        pass

    @property
    def step_counter(self):
        """ Get current step counter. """
        return

    @step_counter.setter
    def step_counter(self, n):
        """ Move step counter to `n` time step. Useful for replay. """
        pass


class Window:
    def __init__(self, name, size, **kwargs):
        """
        :param name: (str) Name of window
        :param size: (tuple[int, int]) Size of window

        kwargs:
        * sim: (GWSim) The simulator who controls the window.
        * title: (str) Window title (Default: "Window")
        * resizable: (bool) Can the window be resized? (Default: False)
        * visible: (bool) Is the window visible? (Default: True) [Note: this minimizes the display.]
        * frame_rate: (float) Frames per second for pygame rendering. (Default: 60)
        * sm_update_rate: (float) State machine updates per second. (Default: 1)
        * backcolor: (tuple[int, int, int]) Default backcolor of window. (Default: (0, 0, 0))

        Programmer's Note:
            * Since SM is a special element, Window handles the SMUPDATE event with sm_update() function.
                Users are not allowed to add any more handlers to this event.
        """
        # Instance variables
        self._sim = None
        self._name = name
        self._controls = dict()
        self._sprites = pygame.sprite.LayeredUpdates()
        self._size = pygame.math.Vector2(*size)
        self._title = kwargs["title"] if "title" in kwargs else f"Window({name})"
        self._backcolor = kwargs["backcolor"] if "backcolor" in kwargs else (0, 0, 0)
        self._resizable = kwargs["resizable"] if "resizable" in kwargs else False
        self._frame_rate = kwargs["frame_rate"] if "frame_rate" in kwargs else 60
        self._sm_update_rate = kwargs["sm_update_rate"] if "sm_update_rate" in kwargs else 1
        self._visible = kwargs["visible"] if "visible" in kwargs else True
        self._running = False

        # TODO: Events parameters
        self._mouse_up_ticks = [-1, -1]
        self._mouse_down_ticks = [-1, -1]
        self._handlers = {
            # Pygame inbuilt events
            pygame.QUIT: [self._on_exit],
            pygame.WINDOWRESIZED: [self._on_window_resized],
            pygame.WINDOWMINIMIZED: [],
            pygame.WINDOWMAXIMIZED: [],
            pygame.WINDOWENTER: [],
            pygame.WINDOWLEAVE: [],
            pygame.WINDOWFOCUSGAINED: [],
            pygame.WINDOWFOCUSLOST: [],
            pygame.WINDOWMOVED: [],
            pygame.WINDOWCLOSE: [],
            pygame.KEYDOWN: [],
            pygame.KEYUP: [],
            pygame.MOUSEBUTTONUP: [],
            pygame.MOUSEBUTTONDOWN: [],
            # Custom events
            (GWSIM_EVENTS, GWSIM_EVENTS_ID["E_WINDOWRESIZABILITYCHANGED"]): [self._on_window_resized],
        }
        self._events = set(self._handlers.keys())

    # ============================================================================================
    # PUBLIC METHODS
    # ============================================================================================
    def add_control(self, control):
        self._controls[control.name] = control
        self._sprites.add(control)

    def rem_control(self, control):
        # Remove control, if exists, from controls list.
        if isinstance(control, str):
            control = self._controls.pop(control, None)
        else:
            control = self._controls.pop(control.name, None)

        # Remove the control from sprite group, if exists.
        if control is not None:
            self._sprites.remove(control)

    def add_event(self, event_id, default_handler=None):
        if event_id not in self._events:
            self._events.add(event_id)
            if default_handler is not None:
                self._handlers.update({event_id: [default_handler]})
            else:
                self._handlers.update({event_id: []})

    def register_handler(self, event_id, handler):
        self._handlers[event_id].append(handler)

    def run(self):
        # Initialize pygame
        pygame.init()

        # Set window parameters
        pygame.display.set_caption(self._title)
        try:
            pygame.display.set_icon(pygame.image.load("sprites/GWSim.png"))
        except FileNotFoundError:
            pass
        screen = pygame.display.set_mode([self.width, self.height], pygame.RESIZABLE)

        # Clock and timer related stuff
        clock = pygame.time.Clock()
        pygame.time.set_timer(
            pygame.event.Event(
                E_SM_UPDATE,
                id=0,
                name="E_SM_UPDATE",
                sender=self.name,
                update_rate=self._sm_update_rate),
            int(self._sm_update_rate * 1000)
        )

        # Start rendering loop
        self._running = True
        while self._running:
            # Update state machine
            for event in pygame.event.get(eventtype=E_SM_UPDATE):
                self.sm_update()

            # Handle events
            self.process_window_events()
            self.trigger_gwsim_events()
            self.process_gwsim_events()

            # Update screen
            self.render_update(screen)

            # Control FPS
            clock.tick(self._frame_rate)

    # ============================================================================================
    # EVENT HANDLERS
    # ============================================================================================
    def sm_update(self):
        print(f"Called: {self}.{inspect.stack()[0][3]}")

    # def process_events(self, ):
    #     # print(f"Called: {self}.{inspect.stack()[0][3]}")
    #
    #     # Handle window events
    #     for event in pygame.event.get():
    #         # For all window events, the sender is self.
    #         sender = self
    #         # Since state machine is a special element of Window, we handle it separately.
    #         if event.type == GWSIM_EVENTS and event.name == E_SMUPDATE:
    #             self.sm_update()
    #
    #         if event.type == GWSIM_EVENTS and event.name == E_NAME_WINDOWRESIZABILITYCHANGED:
    #             for handler in self._handlers[E_NAME_QUIT]:
    #                 handler(sender, {"width": self.width, "height": self.height})
    #
    #         if event.type == pygame.QUIT:
    #             for handler in self._handlers[E_NAME_QUIT]:
    #                 handler(sender, dict())
    #
    #         if event.type == pygame.WINDOWRESIZED:
    #             for handler in self._handlers[E_NAME_WINDOWRESIZED]:
    #                 handler(sender, {"width": event.x, "height": event.y})
    #
    #         if event.type == pygame.WINDOWMINIMIZED:
    #             for handler in self._handlers[E_NAME_WINDOWMINIMIZED]:
    #                 handler(sender, dict())
    #
    #         if event.type == pygame.WINDOWMAXIMIZED:
    #             for handler in self._handlers[E_NAME_WINDOWMAXIMIZED]:
    #                 handler(sender, dict())
    #
    #         if event.type == pygame.WINDOWENTER:
    #             for handler in self._handlers[E_NAME_WINDOWENTER]:
    #                 handler(sender, dict())
    #
    #         if event.type == pygame.WINDOWLEAVE:
    #             for handler in self._handlers[E_NAME_WINDOWLEAVE]:
    #                 handler(sender, dict())
    #
    #         if event.type == pygame.WINDOWFOCUSGAINED:
    #             for handler in self._handlers[E_NAME_WINDOWFOCUSGAINED]:
    #                 handler(sender, dict())
    #
    #         if event.type == pygame.WINDOWFOCUSLOST:
    #             for handler in self._handlers[E_NAME_WINDOWFOCUSLOST]:
    #                 handler(sender, dict())
    #
    #         if event.type == pygame.KEYDOWN:
    #             for handler in self._handlers[E_NAME_KEYDOWN]:
    #                 handler(sender, {"key": event.key, "modifiers": event.mod})
    #
    #         if event.type == pygame.KEYUP:
    #             for handler in self._handlers[E_NAME_KEYUP]:
    #                 handler(sender, {"key": event.key, "modifiers": event.mod})
    #
    #         if event.type == pygame.WINDOWMOVED:
    #             for handler in self._handlers[E_NAME_WINDOWMOVED]:
    #                 handler(sender, {"key": event.key, "modifiers": event.mod})
    #
    #         if event.type == pygame.WINDOWCLOSE:
    #             for handler in self._handlers[E_NAME_WINDOWCLOSE]:
    #                 handler(sender, {"key": event.key, "modifiers": event.mod})
    #
    #     # Trigger events for all controls in window.
    #     for name, control in self._controls.items():
    #         # For event in control.events: Trigger it, if applicable.
    #         control.process_event(event)
    def process_window_events(self):
        # print(f"Called: {self}.{inspect.stack()[0][3]}")
        # For all window events, the sender is self.
        sender = self

        # Since state machine is a special element of Window, we handle it separately.
        for event in pygame.event.get(eventtype=[
            pygame.QUIT,
            pygame.WINDOWRESIZED,
            pygame.WINDOWMINIMIZED,
            pygame.WINDOWMAXIMIZED,
            pygame.WINDOWENTER,
            pygame.WINDOWLEAVE,
            pygame.WINDOWFOCUSGAINED,
            pygame.WINDOWFOCUSLOST,
            pygame.WINDOWMOVED,
            pygame.WINDOWCLOSE,
        ]):

            if event.type == pygame.QUIT:
                for handler in self._handlers[pygame.QUIT]:
                    handler(sender, dict())

            if event.type == pygame.WINDOWRESIZED:
                for handler in self._handlers[pygame.WINDOWRESIZED]:
                    handler(sender, {"width": event.x, "height": event.y})

            if event.type == pygame.WINDOWMINIMIZED:
                for handler in self._handlers[pygame.WINDOWMINIMIZED]:
                    handler(sender, dict())

            if event.type == pygame.WINDOWMAXIMIZED:
                for handler in self._handlers[pygame.WINDOWMAXIMIZED]:
                    handler(sender, dict())

            if event.type == pygame.WINDOWENTER:
                for handler in self._handlers[pygame.WINDOWENTER]:
                    handler(sender, dict())

            if event.type == pygame.WINDOWLEAVE:
                for handler in self._handlers[pygame.WINDOWLEAVE]:
                    handler(sender, dict())

            if event.type == pygame.WINDOWFOCUSGAINED:
                for handler in self._handlers[pygame.WINDOWFOCUSGAINED]:
                    handler(sender, dict())

            if event.type == pygame.WINDOWFOCUSLOST:
                for handler in self._handlers[pygame.WINDOWFOCUSLOST]:
                    handler(sender, dict())

            if event.type == pygame.WINDOWMOVED:
                for handler in self._handlers[pygame.WINDOWMOVED]:
                    handler(sender, dict())

            if event.type == pygame.WINDOWCLOSE:
                for handler in self._handlers[pygame.WINDOWCLOSE]:
                    handler(sender, dict())

    def trigger_gwsim_events(self):
        """
        These events carry the name of control which triggered the event.
        """
        pressed_keys = pygame.key.get_pressed()
        # If any key is pressed, raise keypress event.

        # for event in pygame.event.get(exclude=GWSIM_EVENTS):
        for event in pygame.event.get(eventtype=[
            pygame.KEYDOWN,
            pygame.KEYUP,
            pygame.MOUSEBUTTONDOWN,
            pygame.MOUSEBUTTONUP,
        ]):

            # Handle all events for window
            if event.type == pygame.KEYDOWN:
                for handler in self._handlers[pygame.KEYDOWN]:
                    handler(self, dict())

            if event.type == pygame.KEYUP:
                for handler in self._handlers[pygame.KEYUP]:
                    handler(self, dict())

            if event.type == pygame.MOUSEBUTTONDOWN:
                for handler in self._handlers[pygame.MOUSEBUTTONDOWN]:
                    handler(self, dict())

            if event.type == pygame.MOUSEBUTTONUP:
                for handler in self._handlers[pygame.MOUSEBUTTONUP]:
                    handler(self, dict())

            for name, control in self._controls:
                # control.handles() checks if event is handled by the control?
                if not control.handles(event):
                    continue

                # -------------------------------------------------------------------------------
                # KEYBOARD EVENTS
                # -------------------------------------------------------------------------------
                if event.type == pygame.KEYDOWN:
                    pygame.event.post(
                        pygame.event.Event(
                            GWSIM_EVENTS,
                            id=GWSIM_EVENTS_ID["E_KEYDOWN"],
                            name="E_KEYDOWN",
                            sender=name,
                            key=event.key,
                            mod=event.mod,
                            unicode=event.unicode,
                            scancode=event.scancode
                        )
                    )

                if event.type == pygame.KEYUP:
                    pygame.event.post(
                        pygame.event.Event(
                            GWSIM_EVENTS,
                            id=GWSIM_EVENTS_ID["E_KEYUP"],
                            name="E_KEYUP",
                            sender=name,
                            key=event.key,
                            mod=event.mod,
                            unicode=event.unicode,
                            scancode=event.scancode
                        )
                    )

                # -------------------------------------------------------------------------------
                # MOUSE EVENTS
                # -------------------------------------------------------------------------------
                mouse_position = pygame.mouse.get_pos()

                if control.point_in_control(pygame.math.Vector2(*mouse_position)):
                    # Mouse over event
                    pygame.event.post(
                        pygame.event.Event(
                            GWSIM_EVENTS,
                            id=GWSIM_EVENTS_ID["E_MOUSEHOVER"],
                            name="E_MOUSEHOVER",
                            sender=name,
                            mouse_pos=mouse_position
                        )
                    )

                    if event.type == pygame.MOUSEBUTTONDOWN:
                        time_ms = pygame.time.get_ticks()
                        self._mouse_down_ticks.pop(0)
                        self._mouse_down_ticks.append(time_ms)

                        pygame.event.post(
                            pygame.event.Event(
                                GWSIM_EVENTS,
                                id=GWSIM_EVENTS_ID["E_MOUSEBUTTONDOWN"],
                                name="E_MOUSEBUTTONDOWN",
                                sender=name,
                                mouse_pos=mouse_position
                            )
                        )

                    if event.type == pygame.MOUSEBUTTONUP:
                        time_ms = pygame.time.get_ticks()
                        self._mouse_up_ticks.pop(0)
                        self._mouse_up_ticks.append(time_ms)

                        pygame.event.post(
                            pygame.event.Event(
                                GWSIM_EVENTS,
                                id=GWSIM_EVENTS_ID["E_MOUSEBUTTONUP"],
                                name="E_MOUSEBUTTONUP",
                                sender=name,
                                mouse_pos=mouse_position
                            )
                        )

                # -------------------------------------------------------------------------------
                # CUSTOM EVENTS
                # -------------------------------------------------------------------------------
                if self._mouse_up_ticks[1] - self._mouse_down_ticks[1] > 0 and \
                        pygame.time.get_ticks() - self._mouse_up_ticks[1] < 1000 / self._frame_rate:
                    pygame.event.post(
                        pygame.event.Event(
                            GWSIM_EVENTS,
                            id=GWSIM_EVENTS_ID["E_MOUSECLICK"],
                            name="E_MOUSECLICK",
                            sender=name,
                            mouse_pos=mouse_position
                        )
                    )

                if self._mouse_down_ticks[1] - self._mouse_down_ticks[0] < 500 and \
                        self._mouse_up_ticks[1] - self._mouse_up_ticks[0] < 500 and \
                        pygame.time.get_ticks() - self._mouse_up_ticks[1] < 1000 / self._frame_rate:

                    pygame.event.post(
                        pygame.event.Event(
                            GWSIM_EVENTS,
                            id=GWSIM_EVENTS_ID["E_MOUSEDOUBLECLICK"],
                            name="E_MOUSEDOUBLECLICK",
                            sender=name,
                            mouse_pos=mouse_position
                        )
                    )

                # E_RESIZE
                # E_CELLCHANGED
                # E_MOUSEHOVER
                # E_WINDOWRESIZABILITYCHANGED

    def process_gwsim_events(self):
        for event in pygame.event.get():
            if event.type == GWSIM_EVENTS:
                if event.sender == self.name:
                    for handler in self._handlers[(GWSIM_EVENTS, event.id)]:
                        handler(self, event)
                else:
                    control = self._controls[event.name]
                    control.process_event(event)

    def render_update(self, screen):
        # print(f"Called: {self}.{inspect.stack()[0][3]}")
        if self.resizable:
            screen = pygame.display.set_mode(self.size, pygame.RESIZABLE)
        else:
            screen = pygame.display.set_mode(self.size)

        # Clear previous drawing
        screen.fill(self._backcolor)

        # Update all controls (sprites)
        self._sprites.update()
        self._sprites.draw(screen)

        # Update screen
        pygame.display.flip()

    def get_event_handlers(self, event):
        """ Gets the handlers for the given event. """
        pass

    # ============================================================================================
    # PROPERTIES
    # ============================================================================================
    @property
    def name(self):
        return self._name

    @property
    def events(self):
        return self._events

    @property
    def gwsim(self):
        return self._sim

    @gwsim.setter
    def gwsim(self, sim):
        assert isinstance(sim, GWSim)
        self._sim = sim

    @property
    def controls(self):
        return self._controls

    @property
    def title(self):
        return self._title

    @title.setter
    def title(self, value):
        self._title = value

    @property
    def size(self):
        return self._size

    @size.setter
    def size(self, value: pygame.math.Vector2):
        self._size = value

    @property
    def width(self):
        return self._size[0]

    @width.setter
    def width(self, value):
        raise NotImplementedError("TODO. Raise resize() event.")

    @property
    def height(self):
        return self._size[1]

    @height.setter
    def height(self, value):
        raise NotImplementedError("TODO. Raise resize() event.")

    @property
    def resizable(self):
        return self._resizable

    @resizable.setter
    def resizable(self, value):
        self._resizable = value
        pygame.event.post(
            pygame.event.Event(
                GWSIM_EVENTS,
                id=GWSIM_EVENTS_ID["E_WINDOWRESIZABILITYCHANGED"],
                name="E_WINDOWRESIZABILITYCHANGED",
                sender=self,
                size=self._size
            )
        )

    @property
    def backcolor(self):
        return self._backcolor

    @backcolor.setter
    def backcolor(self, value):
        self._backcolor = value

    @property
    def frame_rate(self):
        return self._frame_rate

    @frame_rate.setter
    def frame_rate(self, value):
        self._frame_rate = value

    @property
    def sm_update_rate(self):
        return self._sm_update_rate

    @sm_update_rate.setter
    def sm_update_rate(self, value):
        self._sm_update_rate = value

    # ============================================================================================
    # DEFAULT EVENT HANDLERS
    # ============================================================================================
    def _on_exit(self, sender, event_args, *args, **kwargs):
        print(f"Called: {self}.{inspect.stack()[0][3]}")
        self._running = False

    def _on_window_resized(self, sender, event_args, *args, **kwargs):
        print(f"Called: {self}.{inspect.stack()[0][3]}")
        if self.resizable:
            self._size = pygame.math.Vector2(event_args["width"], event_args["height"])


class GWSim(StateMachine):
    """
    Is a collection of
    * State machine: All windows display something based on the same state machine.
    * Windows: List of windows.
    """
    def __init__(self, graph, window, **kwargs):
        """
        kwargs:
            * `len_history`: Maximum length of history to store. (Default: float("inf"))
            * `init_state`: Initial state. (Default: None)
            * `main_window`: Name of the main window. (Default: windows[0].name)

        Notes:
            * main_window determines the GWSim's stepping and speed etc.
        """
        super(GWSim, self).__init__(graph)

        # Initialize windows
        assert isinstance(window, Window), "Window must be an instance of Windows."
        self._windows = {window.name: window}
        self._main_window = window.name
        for window in self._windows.values():
            window.gwsim = self

    def __getitem__(self, name):
        """ Gets the window corresponding to the name. """
        return self._windows[name]

    def add_window(self, window: Window):
        self._windows[window.name] = window

    def rem_window(self, window: Window):
        self._windows.pop(window.name, None)

    # def step_forward(self, act, choice_function=None, *args, **kwargs):
    #     """ One-step forward. """
    #     pass
    #
    # def step_backward(self, n):
    #     """ Step backward by `n` steps. """
    #     pass

    def run(self):
        self._windows[self._main_window].run()


class GWSimMultiWindow(StateMachine):
    """
    Is a collection of
    * State machine: All windows display something based on the same state machine.
    * Windows: List of windows.
    """
    def __init__(self, graph, windows, **kwargs):
        """
        kwargs:
            * `len_history`: Maximum length of history to store. (Default: float("inf"))
            * `init_state`: Initial state. (Default: None)
            * `main_window`: Name of the main window. (Default: windows[0].name)

        Notes:
            * main_window determines the GWSim's stepping and speed etc.
        """
        super(GWSim, self).__init__(graph)

        # Initialize windows
        assert all(isinstance(window, Window) for window in windows), "Windows must be an iterable of Windows."
        self._windows = {window.name: window for window in windows}
        self._main_window = kwargs["main_window"] if "main_window" in kwargs else windows[0].name
        for window in self._windows.values():
            window.gw_sim = self

    def __getitem__(self, name):
        """ Gets the window corresponding to the name. """
        return self._windows[name]

    def add_window(self, window: Window):
        self._windows[window.name] = window

    def rem_window(self, window: Window):
        self._windows.pop(window.name, None)

    def step_forward(self, act):
        """ One-step forward. """
        pass

    def step_backward(self, n):
        """ Step backward by `n` steps. """
        pass

    def run(self):
        processes = dict()

        for name, window in self._windows.items():
            process = Process(target=window.run)
            process.daemon = True
            processes[name] = process

        for _, process in processes.items():
            process.start()

        while True:
            # Check if we should terminate processes.
            if not processes[self._main_window].is_alive():
                print(util.ColoredMsg.ok(f"[INFO] Main window closed."))
                for name, process in processes.items():
                    if process.is_alive():
                        process.kill()
                        print(util.ColoredMsg.ok(f"[INFO] Closing window: {name}."))
                break

            # TODO. Process messages and events.

    @property
    def step_counter(self):
        """ Get current step counter. """
        return

    @step_counter.setter
    def step_counter(self, n):
        """ Move step counter to `n` time step. Useful for replay. """
        pass

    @property
    def curr_state(self):
        """ Gets the current state of the game. """
        return

    @curr_state.setter
    def curr_state(self, state):
        """ Sets the current state to given state. """
        pass


class Control(pygame.sprite.Sprite):
    def __init__(self, name, parent, position, size, **kwargs):
        """
        :param name: (Hashable object) Unique identifier of the control.
        :param parent: (Window or Control) Parent of the current control.
        :param position: (tuple[int, int] / pygame.math.Vector2)
            Location of top-left point of self w.r.t. parent's top-left point.
        :param size: (tuple[int, int] / pygame.math.Vector2) Size of control.

        kwargs:
            * visible: (bool) Whether control is visible (Default: True)
            * anchor: (AnchorStyle) Whether control is visible (Default: None)
        """
        super(Control, self).__init__()

        # Instance variables
        self._name = name
        self._parent = parent

        self._controls = dict()
        self._register_with_window(self)

        # Geometry properties
        self._anchor = kwargs["anchor"] if "anchor" in kwargs else AnchorStyle.NONE
        self._position = pygame.math.Vector2(*position)
        self._size = pygame.math.Vector2(*size)
        self._image = pygame.Surface(self._size, flags=pygame.SRCALPHA)
        self._rect = self.image.get_rect()
        self._rect.topleft = self.point_to_world(position)
        self._level = kwargs["level"] if "level" in kwargs else \
            (self._parent.level + 1 if isinstance(self._parent, Control) else 0)

        # UI properties
        self._visible = kwargs["visible"] if "visible" in kwargs else True
        self._backcolor = kwargs["backcolor"] if "backcolor" in kwargs else self._parent.backcolor
        self._backimage = kwargs["backimage"] if "backimage" in kwargs else None
        self._borderstyle = kwargs["borderstyle"] if "borderstyle" in kwargs else BorderStyle.SOLID
        self._bordercolor = kwargs["bordercolor"] if "bordercolor" in kwargs else (0, 0, 0)
        self._borderwidth = kwargs["borderwidth"] if "borderwidth" in kwargs else 1
        self._canselect = kwargs["canselect"] if "canselect" in kwargs else False
        self._is_selected = kwargs["is_selected"] if "is_selected" in kwargs else False

        # Event properties
        self._events = {
            E_NAME_KEYPRESS: None,
            E_NAME_KEYDOWN: None,
            E_NAME_KEYUP: None,
            E_NAME_RESIZE: None,
            E_NAME_MOUSEMOTION: self._is_point_in_control,
            E_NAME_MOUSEBUTTONDOWN: self._is_point_in_control,
            E_NAME_MOUSEBUTTONUP: self._is_point_in_control,
            E_NAME_MOUSEBUTTONPRESS: self._is_point_in_control,
            E_NAME_MOUSEWHEEL: self._is_point_in_control,
        }

        self._handlers = {
            E_NAME_KEYPRESS: [],
            E_NAME_KEYDOWN: [],
            E_NAME_KEYUP: [],
            E_NAME_RESIZE: [],
            E_NAME_MOUSEMOTION: [],
            E_NAME_MOUSEBUTTONDOWN: [],
            E_NAME_MOUSEBUTTONUP: [],
            E_NAME_MOUSEBUTTONPRESS: [],
            E_NAME_MOUSEWHEEL: [],
        }

    def __del__(self):
        self._unregister_with_window(self)

    def __str__(self):
        return f"<{self.__class__.__name__} name={self.name}>"

    def _register_with_window(self, control):
        if isinstance(self._parent, Window):
            self._parent.add_control(control)

        if isinstance(self._parent, Control):
            self._parent._register_with_window(control)

    def _unregister_with_window(self, control):
        if isinstance(self._parent, Window):
            self._parent.rem_control(control)

        if isinstance(self._parent, Control):
            self._parent._unregister_with_window(control)

    def delta(self):
        pass

    def update(self):
        # Update position and size
        # TODO. Resize surface, if applicable.
        self._rect.topleft = self.point_to_world(self.position)

        # If control is not visible, then none of its children are visible either.
        if self.visible:
            # Fill with backcolor, backimage
            self._image.fill(self._backcolor)
            if self._backimage is not None:  # FIXME. Check if this code works.
                self._image.blit(self._backimage, (0, 0))

            # Update borders
            if self._borderstyle == BorderStyle.SOLID:
                pygame.draw.rect(
                    self._image,
                    self._backcolor,
                    pygame.Rect(0, 0, self.rect.width, self.rect.height),
                    self._borderwidth
                )
            else:  # self._borderstyle == BorderStyle.HIDDEN:
                pass

    def process_event(self, event):
        # Get mouse position relative to window
        mouse_position = pygame.mouse.get_pos()

        # Recall that self.rect (x, y, w, h) are defined w.r.t. window.
        if self.rect.collidepoint(mouse_position):
            self.on_mouse_hover(mouse_position)

            # Event: mouse down, up, click
            if event.type == pygame.MOUSEBUTTONDOWN:
                self.on_mouse_down(mouse_position)
                self.on_mouse_click(mouse_position)

            if event.type == pygame.MOUSEBUTTONUP:
                self.on_mouse_up(mouse_position)

        # Event: key pressed
        keys = pygame.key.get_pressed()
        if any(key for key in keys):
            self.on_key_press(keys)

        # Event: key down
        if event.type == pygame.KEYDOWN:
            # FIXME: Do we need to handle modifier keys with non-modifier keys?
            key_name = pygame.key.name(event.key)
            if key_name not in self._keydown_pressed_keys:
                self._keydown_pressed_keys.add(key_name)
                self.on_key_down(event)

        # Event: key up
        if event.type == pygame.KEYUP:
            key_name = pygame.key.name(event.key)
            if key_name in self._keydown_pressed_keys:
                self._keydown_pressed_keys.remove(key_name)
                self.on_key_up(event)

    def show(self):
        _past_visibility = self._visible
        self._visible = True
        if _past_visibility != self._visible:
            self.on_visible_changed(self._visible)

    def hide(self):
        _past_visibility = self._visible
        self._visible = False
        if _past_visibility != self._visible:
            self.on_visible_changed(self._visible)

    def scale_controls(self, scale=1):
        raise NotImplementedError("Will be implemented in future.")

    def draw_to_png(self, filename):
        pass

    def point_to_local(self, world_point: pygame.math.Vector2):
        if isinstance(self._parent, Window):
            return world_point
        parent_topleft_world = self._parent.point_to_world(self._parent.position)
        # return world_point[0] - world_parent_topleft[0], world_point[1] - world_parent_topleft[1]
        return world_point - parent_topleft_world

    def point_to_world(self, control_point: pygame.math.Vector2):
        if isinstance(self._parent, Window):
            return control_point
        # return self._parent.world_position[0] + control_point[0], self._parent.world_position[1] + control_point[1]
        return self.parent.point_to_world(self.parent.position) + control_point

    def get_mouse_position(self):
        # TODO. with Events.
        raise NotImplementedError("TODO with Events.")
        # world_position = self._parent.get_mouse_position()
        # return self.point_to_local(world_position)

    def add_control(self, control):
        self._controls[control.name] = control

    def rem_control(self, control):
        # Remove control, if exists, from controls list.
        if isinstance(control, str):
            control = self._controls.pop(control, None)
        else:
            control = self._controls.pop(control.name, None)

    def create_control(self, cls_control, constructor_kwargs):
        # Preprocess input arguments (basic control arguments, any addtional parameters should be passed by user)
        assert "name" in constructor_kwargs, "constructor_kwargs must have 'name' parameter."
        assert "size" in constructor_kwargs, "constructor_kwargs must have 'size' parameter."
        constructor_kwargs["parent"] = self
        constructor_kwargs["position"] = constructor_kwargs["position"] if "position" in constructor_kwargs else (0, 0)

        # Construct control
        control = cls_control(**constructor_kwargs)

        # Add control to window
        self.add_control(control)

    def move_by(self, vec: pygame.math.Vector2):
        self.position = self.position + vec
        # self._rect.left += dx
        # self._rect.top += dy
        for control in self._controls.values():
            control.move_by(vec)

    def move_up_by(self, dy):
        dy = abs(dy)
        self.move_by(pygame.math.Vector2(0, -dy))

    def move_down_by(self, dy):
        dy = abs(dy)
        self.move_by(pygame.math.Vector2(0, dy))

    def move_left_by(self, dx):
        dx = abs(dx)
        self.move_by(pygame.math.Vector2(-dx, 0))

    def move_right_by(self, dx):
        dx = abs(dx)
        self.move_by(pygame.math.Vector2(dx, 0))

    # ===========================================================================
    # PROPERTIES
    # ===========================================================================
    @property
    def image(self):
        # print(f"Call: {self.name}.image")
        return self._image

    @property
    def rect(self):
        # print(f"Call: {self.name}.image")
        return self._rect

    @property
    def level(self):
        # print(f"Call: {self.name}.image")
        return self._level

    @property
    def controls(self):
        return self._controls

    @property
    def name(self):
        return self._name

    @property
    def world_position(self):
        return self.point_to_world(self.position)

    @property
    def parent(self):
        return self._parent

    @parent.setter
    def parent(self, value):
        self._parent = value

    @property
    def position(self):
        """ Gets the location of top-left point of rectangle w.r.t. parent. """
        if self._anchor == AnchorStyle.NONE:
            return self._position
        elif self._anchor == AnchorStyle.TOP_LEFT:
            return pygame.math.Vector2(0, 0)
        elif self._anchor == AnchorStyle.CENTER:
            parent_size = self.parent.size
            self_size = self.size
            return pygame.math.Vector2(0.5 * (parent_size - self_size))
        else:
            raise NotImplementedError(f"Unsupported AnchorStyle: {self._anchor}")

    @position.setter
    def position(self, value):
        """ Gets the location of top-left point of rectangle w.r.t. parent. """
        self._position = value

    @property
    def size(self):
        return self._size

    @size.setter
    def size(self, value: pygame.math.Vector2):
        self._size = value

    @property
    def left(self):
        return self.position[0]

    @left.setter
    def left(self, value):
        self.position = pygame.math.Vector2(value, self.position[1])

    @property
    def top(self):
        return self.position[1]

    @top.setter
    def top(self, value):
        self.position = pygame.math.Vector2(self.position[0], value)

    @property
    def width(self):
        return self._size[0]

    @width.setter
    def width(self, value):
        self.size = pygame.math.Vector2(value, self.position[1])

    @property
    def height(self):
        return self._size[1]

    @height.setter
    def height(self, value):
        self.size = pygame.math.Vector2(self.position[0], value)

    @property
    def visible(self):
        return self._visible

    @visible.setter
    def visible(self, value):
        self._visible = value

    @property
    def backcolor(self):
        return self._backcolor

    @backcolor.setter
    def backcolor(self, value):
        self._backcolor = value

    @property
    def backimage(self):
        return self._backimage

    @backimage.setter
    def backimage(self, value):
        self._backimage = value

    @property
    def borderstyle(self):
        return self._borderstyle

    @borderstyle.setter
    def borderstyle(self, value):
        self._borderstyle = value

    @property
    def borderwidth(self):
        return self._borderwidth

    @borderwidth.setter
    def borderwidth(self, value):
        self._borderwidth = value

    @property
    def bordercolor(self):
        return self._bordercolor

    @bordercolor.setter
    def bordercolor(self, value):
        self._bordercolor = value

    @property
    def can_select(self):
        return self._canselect

    @can_select.setter
    def can_select(self, value):
        self.can_select = value

    # ===========================================================================
    # EVENTS
    # ===========================================================================
    def on_mouse_click(self, event_args):
        pass

    def on_mouse_hover(self, event_args):
        pass

    def on_mouse_enter(self, event_args):
        pass

    def on_mouse_leave(self, event_args):
        pass

    def on_mouse_move(self, event_args):
        pass

    def on_mouse_down(self, event_args):
        pass

    def on_mouse_up(self, event_args):
        pass

    def on_mouse_wheel(self, event_args):
        pass

    def on_key_down(self, event_args):
        pass

    def on_key_up(self, event_args):
        pass

    def on_key_press(self, event_args):
        pass

    def on_control_added(self, event_args):
        pass

    def on_control_removed(self, event_args):
        pass

    def on_visible_changed(self, event_args):
        pass

    def on_selected(self, event_args):
        pass

    def on_unselected(self, event_args):
        pass

    def on_control_moved(self, event_args):
        x, y = event_args

        if x < 0:
            x = 0

        if x + self.width > self._parent.width:
            x = self._parent.width - self.width

        if y < 0:
            y = 0

        if y + self.height > self._parent.height:
            y = self._parent.height - self.height

        self._position = [x, y]

    def _is_point_in_control(self, vec: pygame.math.Vector2):
        return True


class Grid(Control):
    def __init__(self, name, parent, position, size, grid_size, **kwargs):
        """
        Special kwargs:
        * cls_cell: (Cell) The class (Cell or derived from Cell) that is used to construct background cells in grid.
        """
        super(Grid, self).__init__(name, parent, position, size, **kwargs)

        # Grid
        self._grid_size = grid_size
        self._controls = dict()
        self._sprites = {
            (x, y): pygame.sprite.LayeredUpdates() for x in range(grid_size[0]) for y in range(grid_size[1])
        }
        self._grid_layout = kwargs["grid_layout"] if "grid_layout" in kwargs else GridLayout.AUTO
        self._cls_cell = kwargs["cls_cell"] if "cls_cell" in kwargs else Cell

        if self._grid_layout == GridLayout.AUTO:
            self._construct_grid(self._cls_cell)
        elif self._grid_layout == GridLayout.CUSTOM:
            self.construct_grid()
        else:
            raise ValueError("GridLayout unrecognized.")

    def __getitem__(self, cell):
        return self._controls[cell]

    def _construct_grid(self, cls_cell):
        """ Auto grid construction. Uniform cells. """
        rows, cols = self._grid_size
        cell_size = (self.width // rows, self.height // cols)
        for x in range(rows):
            for y in range(cols):
                position = (cell_size[0] * x, cell_size[1] * y)
                cell_xy = cls_cell(
                    name=(x, y), parent=self, position=position, size=cell_size,
                    bordercolor=self._bordercolor, borderstyle=self._borderstyle, borderwidth=self._borderwidth,
                    level=0
                )
                self._controls[(x, y)] = cell_xy
                self._sprites[(x, y)].add(cell_xy)

    def construct_grid(self):
        raise NotImplementedError("User should implement this if grid is generated in custom mode.")


class Cell(Control):
    def update(self):
        # print(f"Called: {self}.{inspect.stack()[0][3]}")
        super(Cell, self).update()

        if self._borderstyle == BorderStyle.SOLID:
            pygame.draw.rect(
                self.image,
                self._bordercolor,
                pygame.Rect(0, 0, self.rect.width, self.rect.height),
                self._borderwidth
            )
        else:
            self.image.fill(COLOR_TRANSPARENT)


class Character(Control):
    pass