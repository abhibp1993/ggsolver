import inspect
import pygame
import random
import scipy.stats as stats
import ggsolver.gridworld.color_util as colors

# ===========================================================================================
# GLOBALS
# ===========================================================================================
GWSIM_EVENTS = pygame.USEREVENT + 1
GWSIM_EVENTS_SM_UPDATE = 0
GWSIM_EVENTS_GRIDCELL_ENTER = 1
GWSIM_EVENTS_GRIDCELL_LEAVE = 2


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


class DockStyle:
    NONE = "None"
    TOP_LEFT = "Top-left"
    TOP_RIGHT = "Top-right"
    BOTTOM_LEFT = "Bottom-left"
    BOTTOM_RIGHT = "Bottom-right"
    CENTER = "Center"


# ===========================================================================================
# SIMULATION OBJECTS
# ===========================================================================================
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
        if isinstance(choices, stats.rv_discrete):
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
        * on_quit: (function[event_args: pygame.event.Event] -> None) Handler for pygame.QUIT event. (Default: None)
        * on_window_resized: (function[event_args: pygame.event.Event] -> None) Handler for pygame.WINDOWRESIZED event. (Default: None)
        * on_window_minimized: (function[event_args: pygame.event.Event] -> None) Handler for pygame.WINDOWMINIMIZED event. (Default: None)
        * on_window_maximized: (function[event_args: pygame.event.Event] -> None) Handler for pygame.WINDOWMAXIMIZED event. (Default: None)
        * on_window_enter: (function[event_args: pygame.event.Event] -> None) Handler for pygame.WINDOWENTER event. (Default: None)
        * on_window_leave: (function[event_args: pygame.event.Event] -> None) Handler for pygame.WINDOWLEAVE event. (Default: None)
        * on_window_focus_gained: (function[event_args: pygame.event.Event] -> None) Handler for pygame.WINDOWFOCUSGAINED event. (Default: None)
        * on_window_focus_lost: (function[event_args: pygame.event.Event] -> None) Handler for pygame.WINDOWFOCUSLOST event. (Default: None)
        * on_window_moved: (function[event_args: pygame.event.Event] -> None) Handler for pygame.WINDOWMOVED event. (Default: None)
        * on_window_close: (function[event_args: pygame.event.Event] -> None) Handler for pygame.WINDOWCLOSE event. (Default: None)
        * on_key_up: (function[event_args: pygame.event.Event] -> None) Handler for pygame.KEYUP event. (Default: None)
        * on_key_down: (function[event_args: pygame.event.Event] -> None) Handler for pygame.KEYDOWN event. (Default: None)
        * on_mouse_button_up: (function[event_args: pygame.event.Event] -> None) Handler for pygame.MOUSEBUTTONUP event. (Default: None)
        * on_mouse_button_down: (function[event_args: pygame.event.Event] -> None) Handler for pygame.MOUSEBUTTONDOWN event. (Default: None)

        Programmer's Note:
            * Since SM is a special element, Window handles the SMUPDATE event with sm_update() function.
                Users are not allowed to add any more handlers to this event.

        SPECIAL KEY MAPPING:
            * SHIFT + P: Pause or Unpause game. (see self._on_key_down function)
        """
        # Instance variables
        self._gw_sim = None
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
        self._game_paused = False

        # Event handling flags
        self._e_flag_on_window_resized = False

        # Event handlers
        self._event_handlers = dict()
        self._initialize_event_handlers(**kwargs)

    # ============================================================================================
    # PROPERTIES
    # ============================================================================================
    @property
    def name(self):
        return self._name

    @property
    def events(self):
        return list(self._event_handlers.keys())

    @property
    def gw_sim(self):
        return self._gw_sim

    @gw_sim.setter
    def gw_sim(self, sim):
        assert isinstance(sim, GWSim)
        self._gw_sim = sim

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
        self._e_flag_on_window_resized = True

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
                GWSIM_EVENTS,
                id=GWSIM_EVENTS_SM_UPDATE,
                sender=self,
                update_rate=self._sm_update_rate),
            int(self._sm_update_rate * 1000)
        )

        # Start rendering loop
        self._running = True
        while self._running:
            # Trigger custom events
            self.trigger_custom_events()

            # Handle all events
            for event in pygame.event.get():
                self.process_event(event)

            # Update screen
            self.render_update(screen)

            # Control FPS
            clock.tick(self._frame_rate)

    def sm_update(self, sender, event_args):
        print(f"Called: {self}.{inspect.stack()[0][3]}")

    def trigger_custom_events(self):
        if self._e_flag_on_window_resized:
            pygame.event.post(
                pygame.event.Event(
                    pygame.WINDOWRESIZED,
                    trigger=self,
                    x=self.width,
                    y=self.height
                )
            )

    def process_event(self, event):
        # Handle in-built pygame events
        if event.type < pygame.USEREVENT and event.type in self._event_handlers.keys():
            sender = self
            for func in self._event_handlers[event.type]:
                func(sender=sender, event_args=event)

        # Handle custom GWSim events
        if event.type == GWSIM_EVENTS and (event.type, event.id) in self._event_handlers.keys():
            if event.id == GWSIM_EVENTS_SM_UPDATE and not self._game_paused:
                sender = self
                for func in self._event_handlers[(event.type, event.id)]:
                    func(sender=sender, event_args=event)

        # Trigger events for registered controls
        for control in self._controls.values():
            # sender = control
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

    # =================================================================================
    # EVENT CONFIGURATION
    # =================================================================================
    def add_event_handler(self, event_id, func):
        """
        Adds an event handler to a window event.

        :param event_id: If event is built-in pygame.event then event_id is its type.
            If event is GWSimEvent, then event_id is a tuple (GWSIM_EVENTS, gwsim_event_id)
        :param func: (function[dict] -> None)
        """
        if event_id in self._event_handlers.keys():
            self._event_handlers[event_id].append(func)

    def get_handlers(self, event):
        # If event is in-built pygame event
        if event.type < pygame.USEREVENT and event.type in self._event_handlers.keys():
            return self._event_handlers[event.type]
        elif (event.type, event.id) in self._event_handlers.keys():
            return self._event_handlers[(event.type, event.id)]
        else:
            return []

    def _initialize_event_handlers(self, **kwargs):
        """
        Initializes default and user event handlers
        :param kwargs: The following keyword arguments are processed.
            * on_quit: (function[event_args] -> None) Handler for pygame.QUIT event. (Default: None)
            * on_window_resized: (function[event_args] -> None) Handler for pygame.WINDOWRESIZED event. (Default: None)
            * on_window_minimized: (function[event_args] -> None) Handler for pygame.WINDOWMINIMIZED event. (Default: None)
            * on_window_maximized: (function[event_args] -> None) Handler for pygame.WINDOWMAXIMIZED event. (Default: None)
            * on_window_enter: (function[event_args] -> None) Handler for pygame.WINDOWENTER event. (Default: None)
            * on_window_leave: (function[event_args] -> None) Handler for pygame.WINDOWLEAVE event. (Default: None)
            * on_window_focus_gained: (function[event_args] -> None) Handler for pygame.WINDOWFOCUSGAINED event. (Default: None)
            * on_window_focus_lost: (function[event_args] -> None) Handler for pygame.WINDOWFOCUSLOST event. (Default: None)
            * on_window_moved: (function[event_args] -> None) Handler for pygame.WINDOWMOVED event. (Default: None)
            * on_window_close: (function[event_args] -> None) Handler for pygame.WINDOWCLOSE event. (Default: None)

        .. note:: The private handlers are default handlers and are called before executing user handlers.
        """
        self._event_handlers = {
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
            pygame.KEYDOWN: [self._on_key_down],
            pygame.KEYUP: [],
            pygame.MOUSEBUTTONUP: [],
            pygame.MOUSEBUTTONDOWN: [],
            (GWSIM_EVENTS, GWSIM_EVENTS_SM_UPDATE): [self.sm_update]
        }
        if "on_quit" in kwargs:
            self.add_event_handler(pygame.QUIT, kwargs["on_quit"])
        if "on_window_resized" in kwargs:
            self.add_event_handler(pygame.WINDOWRESIZED, kwargs["on_window_resized"])
        if "on_window_minimized" in kwargs:
            self.add_event_handler(pygame.WINDOWMINIMIZED, kwargs["on_window_minimized"])
        if "on_window_maximized" in kwargs:
            self.add_event_handler(pygame.WINDOWMAXIMIZED , kwargs["on_window_maximized"])
        if "on_window_enter" in kwargs:
            self.add_event_handler(pygame.WINDOWENTER, kwargs["on_window_enter"])
        if "on_window_leave" in kwargs:
            self.add_event_handler(pygame.WINDOWLEAVE, kwargs["on_window_leave"])
        if "on_window_focus_gained" in kwargs:
            self.add_event_handler(pygame.WINDOWFOCUSGAINED, kwargs["on_window_focus_gained"])
        if "on_window_focus_lost" in kwargs:
            self.add_event_handler(pygame.WINDOWFOCUSLOST, kwargs["on_window_focus_lost"])
        if "on_window_moved" in kwargs:
            self.add_event_handler(pygame.WINDOWMOVED, kwargs["on_window_moved"])
        if "on_window_close" in kwargs:
            self.add_event_handler(pygame.WINDOWCLOSE, kwargs["on_window_close"])
        if "on_key_up" in kwargs:
            self.add_event_handler(pygame.KEYUP, kwargs["on_key_up"])
        if "on_key_down" in kwargs:
            self.add_event_handler(pygame.KEYDOWN, kwargs["on_key_down"])
        if "on_mouse_button_up" in kwargs:
            self.add_event_handler(pygame.MOUSEBUTTONUP, kwargs["on_mouse_button_up"])
        if "on_mouse_button_down" in kwargs:
            self.add_event_handler(pygame.MOUSEBUTTONDOWN, kwargs["on_mouse_button_down"])

    # =================================================================================
    # DEFAULT EVENT HANDLERS
    # =================================================================================
    def _on_exit(self, sender, event_args):
        print(f"Called: {self}.{inspect.stack()[0][3]}")
        self._running = False

    def _on_window_resized(self, sender, event_args):
        print(f"Called: {self}.{inspect.stack()[0][3]}")
        if self.resizable:
            self._size = pygame.math.Vector2(event_args["width"], event_args["height"])

    def _on_key_down(self, sender, event_args):
        mods = pygame.key.get_mods()
        if event_args.key == pygame.K_p and mods & pygame.KMOD_SHIFT:
            self._game_paused = not self._game_paused
            print(f"[INFO] Game {'running.' if not self._game_paused else 'paused.'}")


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

    def run(self):
        self._windows[self._main_window].run()


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
            * dockstyle: (DockStyle) Snap position of self to parent. (Default: DockStyle.NONE)
            * on_key_up: (function[event_args: pygame.event.Event] -> None) Handler for pygame.KEYUP event. (Default: None)
            * on_key_down: (function[event_args: pygame.event.Event] -> None) Handler for pygame.KEYDOWN event. (Default: None)
            * on_mouse_button_up: (function[event_args: pygame.event.Event] -> None) Handler for pygame.MOUSEBUTTONUP event. (Default: None)
            * on_mouse_button_down: (function[event_args: pygame.event.Event] -> None) Handler for pygame.MOUSEBUTTONDOWN event. (Default: None)

        """
        super(Control, self).__init__()

        # Instance variables
        self._name = name
        self._parent = parent

        self._controls = dict()
        self._register_with_window(self)
        if not isinstance(self._parent, Window):
            self._parent.add_control(self)

        # Geometry properties
        self._dockstyle = kwargs["dockstyle"] if "dockstyle" in kwargs else DockStyle.NONE
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

        # Event handlers
        self._event_handlers = dict()
        self._initialize_event_handlers(**kwargs)

    def __del__(self):
        self._unregister_with_window(self)

    def __str__(self):
        return f"<{self.__class__.__name__} name={self.name}>"

    # ============================================================================================
    # PROPERTIES
    # ============================================================================================
    @property
    def image(self):
        # print(f"Call: {self.name}.image")
        return self._image

    @property
    def rect(self):
        # print(f"Call: {self.name}.image")
        # return self._image.get_rect()
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
        self._parent.rem_control(self)
        self._parent = value
        self._parent.add_control(self)

    @property
    def dock(self):
        """ Gets the location of top-left point of rectangle w.r.t. parent. """
        return self._dockstyle

    @dock.setter
    def dock(self, value):
        """ Gets the location of top-left point of rectangle w.r.t. parent. """
        self._dockstyle = value

    @property
    def position(self):
        """ Gets the location of top-left point of rectangle w.r.t. parent. """
        return self._position

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
        self.position = pygame.math.Vector2(value, self.top)

    @property
    def top(self):
        return self.position[1]

    @top.setter
    def top(self, value):
        self.position = pygame.math.Vector2(self.left, value)

    @property
    def width(self):
        return self._size[0]

    @width.setter
    def width(self, value):
        self.size = pygame.math.Vector2(value, self.height)

    @property
    def height(self):
        return self._size[1]

    @height.setter
    def height(self, value):
        self.size = pygame.math.Vector2(self.width, value)

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

    # ============================================================================================
    # EVENT MANAGERS
    # ============================================================================================
    def _initialize_event_handlers(self, **kwargs):
        self._event_handlers = {
            pygame.KEYDOWN: [],
            pygame.KEYUP: [],
            pygame.MOUSEBUTTONUP: [],
            pygame.MOUSEBUTTONDOWN: [],
        }
        if "on_key_up" in kwargs:
            self.add_event_handler(pygame.KEYUP, kwargs["on_key_up"])
        if "on_key_down" in kwargs:
            self.add_event_handler(pygame.KEYDOWN, kwargs["on_key_down"])
        if "on_mouse_button_up" in kwargs:
            self.add_event_handler(pygame.MOUSEBUTTONUP, kwargs["on_mouse_button_up"])
        if "on_mouse_button_down" in kwargs:
            self.add_event_handler(pygame.MOUSEBUTTONDOWN, kwargs["on_mouse_button_down"])

    def add_event_handler(self, event_id, func):
        """
        Adds an event handler to a window event.

        :param event_id: If event is built-in pygame.event then event_id is its type.
            If event is GWSimEvent, then event_id is a tuple (GWSIM_EVENTS, gwsim_event_id)
        :param func: (function[dict] -> None)
        """
        if event_id in self._event_handlers.keys():
            self._event_handlers[event_id].append(func)

    def get_handlers(self, event):
        # If event is in-built pygame event
        if event.type < pygame.USEREVENT and event.type in self._event_handlers.keys():
            return self._event_handlers[event.type]
        elif (event.type, event.id) in self._event_handlers.keys():
            return self._event_handlers[(event.type, event.id)]
        else:
            return []

    def process_event(self, event):
        if not self.handles(event):
            return

        # Handle in-built pygame events
        if event.type < pygame.USEREVENT:
            # print(self, pygame.event.event_name(event.type), self._event_handlers[event.type])
            for func in self._event_handlers[event.type]:
                func(sender=self, event_args=event)

        # Handle custom GWSim events
        if event.type == GWSIM_EVENTS and (event.type, event.id) in self._event_handlers.keys():
            # print(self, (pygame.event.event_name(event.type), event.id), self._event_handlers[(event.type, event.id)])
            for func in self._event_handlers[(event.type, event.id)]:
                func(sender=self, event_args=event)

    def handles(self, event):
        if event.type < pygame.USEREVENT:
            return event.type in self._event_handlers.keys()
        return (event.type, event.id) in self._event_handlers.keys()

    # ============================================================================================
    # DEFAULT EVENT HANDLERS
    # ============================================================================================

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

    # ============================================================================================
    # PUBLIC FUNCTIONS: RENDERING
    # ============================================================================================
    def update(self):
        # Resize
        self._image = pygame.transform.scale(self._image, self.size)
        self._rect = self._image.get_rect()
        
        # Determine position based on DockStyle
        if self._dockstyle == DockStyle.NONE:
            position = self.point_to_world(self.position)
        elif self._dockstyle == DockStyle.TOP_LEFT:
            position = self.point_to_world(
                pygame.math.Vector2([0, 0])
            )
        elif self._dockstyle == DockStyle.TOP_RIGHT:
            position = self.point_to_world(
                pygame.math.Vector2([self.parent.width - self.width, 0])
            )
        elif self._dockstyle == DockStyle.BOTTOM_LEFT:
            position = self.point_to_world(
                pygame.math.Vector2([0, self.parent.height - self.height])
            )
        elif self._dockstyle == DockStyle.BOTTOM_RIGHT:
            position = self.point_to_world(
                pygame.math.Vector2([self.parent.width - self.width, self.parent.height - self.height])
            )
        elif self._dockstyle == DockStyle.CENTER:
            position = self.point_to_world(
                pygame.math.Vector2([(self.parent.width - self.width) / 2, (self.parent.height - self.height) / 2])
            )
        else:
            raise NotImplementedError(f"Unsupported AnchorStyle: {self._dockstyle}")

        # Update rectangle's position
        self._rect.topleft = position

        # If control is not visible, then none of its children are visible either.
        if self.visible:
            # Fill with backcolor, backimage
            self._image.fill(self._backcolor)
            if self._backimage is not None:
                img = pygame.transform.scale(self._backimage, self._rect.size)
                self._image.blit(img, (0, 0))

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
        else:
            # Fill with transperant backcolor
            self._image.fill(colors.COLOR_TRANSPARENT)

    def show(self):
        raise NotImplementedError
        # _past_visibility = self._visible
        # self._visible = True
        # if _past_visibility != self._visible:
        #     self.on_visible_changed(self._visible)

    def hide(self):
        raise NotImplementedError
        # _past_visibility = self._visible
        # self._visible = False
        # if _past_visibility != self._visible:
        #     self.on_visible_changed(self._visible)

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

    def _is_point_in_control(self, vec: pygame.math.Vector2):
        return True

    # ============================================================================================
    # PUBLIC FUNCTIONS: CHILD CONTROLS
    # ============================================================================================
    def add_control(self, control):
        self._controls[control.name] = control

    def rem_control(self, control):
        # Remove control, if exists, from controls list.
        if isinstance(control, str):
            control = self._controls.pop(control, None)
        else:
            control = self._controls.pop(control.name, None)
        return control

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

    # ============================================================================================
    # PUBLIC FUNCTIONS: MOVEMENT
    # ============================================================================================
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


class Grid(Control):
    def __init__(self, name, parent, position, size, grid_size, **kwargs):
        """
        Special kwargs:
        * cls_cell: (Cell) The class (Cell or derived from Cell) that is used to construct background cells in grid.

        Special events: (same handler will be shared with all cells)
        * on_cell_enter: (function[event_args: pygame.event.Event] -> None) Handler for pygame.QUIT event. (Default: None)
        * on_cell_leave: (function[event_args: pygame.event.Event] -> None) Handler for pygame.QUIT event. (Default: None)
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
            self._construct_grid(self._cls_cell, **kwargs)
        elif self._grid_layout == GridLayout.CUSTOM:
            self.construct_grid(**kwargs)
        else:
            raise ValueError("GridLayout unrecognized.")

    def __getitem__(self, cell):
        return self._controls[cell]

    def _construct_grid(self, cls_cell, **kwargs):
        """ Auto grid construction. Uniform cells. """

        # Check for special events (on_cell_leave, on_cell_enter)
        cell_kwargs = {k: v for k, v in kwargs.items() if k in ["on_cell_enter", "on_cell_leave"]}

        rows, cols = self._grid_size
        cell_size = (self.width // rows, self.height // cols)
        for x in range(rows):
            for y in range(cols):
                position = (cell_size[0] * x, self.height - cell_size[1] * (y + 1))
                cell_xy = cls_cell(
                    name=(x, y),
                    parent=self,
                    position=position,
                    size=cell_size,
                    bordercolor=self._bordercolor,
                    borderstyle=self._borderstyle,
                    borderwidth=self._borderwidth,
                    level=0,
                    **cell_kwargs
                )
                self._controls[(x, y)] = cell_xy
                self._sprites[(x, y)].add(cell_xy)

    def construct_grid(self, **kwargs):
        raise NotImplementedError("User should implement this if grid is generated in custom mode.")


class Cell(Control):
    def __init__(self, name, parent, position, size, **kwargs):
        """
        Special Events:
        * on_cell_enter: (function[event_args: pygame.event.Event] -> None) Handler for pygame.QUIT event. (Default: None)
        * on_cell_leave: (function[event_args: pygame.event.Event] -> None) Handler for pygame.QUIT event. (Default: None)
        """
        super(Cell, self).__init__(name, parent, position, size, **kwargs)

        # Special event handlers
        self._event_handlers[(GWSIM_EVENTS, GWSIM_EVENTS_GRIDCELL_ENTER)] = []
        self._event_handlers[(GWSIM_EVENTS, GWSIM_EVENTS_GRIDCELL_LEAVE)] = []

        if "on_cell_enter" in kwargs:
            self.add_event_handler((GWSIM_EVENTS, GWSIM_EVENTS_GRIDCELL_ENTER), kwargs["on_cell_enter"])
        if "on_cell_leave" in kwargs:
            self.add_event_handler((GWSIM_EVENTS, GWSIM_EVENTS_GRIDCELL_LEAVE), kwargs["on_cell_leave"])

    def __repr__(self):
        return f"<{self.__class__.__name__} at name:{self.name}>"

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
            self.image.fill(colors.COLOR_TRANSPARENT)

    def add_control(self, control):
        super(Cell, self).add_control(control)
        try:
            pygame.event.post(
                pygame.event.Event(
                    GWSIM_EVENTS,
                    id=GWSIM_EVENTS_GRIDCELL_ENTER,
                    trigger=self,
                    new_control=control
                )
            )
        except pygame.error:
            pass

    def rem_control(self, control):
        control = super(Cell, self).rem_control(control)
        if control is not None:
            pygame.event.post(
                pygame.event.Event(
                    GWSIM_EVENTS,
                    id=GWSIM_EVENTS_GRIDCELL_LEAVE,
                    trigger=self,
                    rem_control=control
                )
            )
