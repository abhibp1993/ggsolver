"""
Classes:
* Window: base window with communication capabilities.
* Control: base control with event system + surface rendering etc.
* GWSim: Window that runs a state machine + some special key-bindings for mode, stepping etc.
* Grid: gridworld control
* Cell: cell of gridworld (a control)
* Character: animated + sound enabled player and non-player characters
* LogBox (future)
* ListBox (future)

Programmer's notes:
* Nested sprites architecture is not good w.r.t. rendering. It seems pygame is not designed for that :)
    So, let's render all sprites (using some logic) in Window itself.
    Positioning of sprites in gridworld can be tracked separately: self._controls in GWSim.
    A control is just a Sprite with special properties and events.

TODO (features)
* Keep control within bounds of their parents.
* Make Character objects selectable.
* Selected characters show their available actions, and show using color whether they think they are winning or not.
    - Caveat: If P2 is selected with P1's turn, then should P2's perception of P1's action be shown?
* Thought bubble animation. OnClick, thought bubble should zoom in. OnClick again, thought bubble should zoom out.
* Show graphs in a separate window for understanding state of progress made by the agents.
"""

import pygame
from typing import List


COLOR_TRANSPARENT = pygame.Color(0, 0, 0, 0)   # The last 0 indicates 0 alpha, a transparent color


class BorderStyle:
    SOLID = "solid"
    HIDDEN = "hidden"


class GridLayout:
    AUTO = "auto"
    CUSTOM = "custom"


class GameMode:
    AUTO = "auto"
    MANUAL = "manual"


class Window:
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
        # Initialize pygame
        pygame.init()

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

        # Initialize pygame window
        pygame.display.set_caption(self._title)
        try:
            pygame.display.set_icon(pygame.image.load("sprites/GWSim.png"))
        except FileNotFoundError:
            pass
        self._screen = pygame.display.set_mode([self.width, self.height], pygame.RESIZABLE)

        # Add current window to active windows
        Window.ACTIVE_WINDOWS.append(self)

    def __del__(self):
        Window.ACTIVE_WINDOWS.remove(self)

    def delta(self):
        pass

    def update(self):
        # print(f"Called: {self}.{inspect.stack()[0][3]}")
        # Clear previous drawing
        self._screen.fill(self._backcolor)

        # Update all controls (sprites)
        self._sprites.update()
        self._sprites.draw(self._screen)

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
            control.handle_event(event)

    def run(self, args=None):
        clock = pygame.time.Clock()
        self._running = True
        while self._running:
            # Handle handle_event
            for event in pygame.event.get():
                self.handle_event(event)

            # Update window state and sprite visualization update.
            self.delta()
            self.update()

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


class GWSim(Window):
    def __init__(self, name, size, graph, **kwargs):
        """
        graph: (ggsolver.graph.Graph) game graph

        Special kwargs:
        * init_node: (int) Initial node of state machine.
        * step_frequency: (int) Frequency with which the state machine should be stepped.
        * mode: (GameMode) Mode of operation for simulation.
        """
        super(GWSim, self).__init__(name, size, **kwargs)

        # Load game graph
        self._graph = graph
        self._state_to_node = dict()
        self._cache_state_to_node()

        # State machine
        self._curr_state = None
        self._state_history = []
        self._action_history = []
        self._memory_limit = float("inf")
        self._time_step = 0
        self._time_reversed = False

    def _cache_state_to_node(self):
        pass

    # TODO. Add properties about state machine.


class Control(pygame.sprite.Sprite):
    def __init__(self, name, parent, position, size, **kwargs):
        """
        :param name:
        kwargs:
        * visible: (bool) Whether control is visible (Default: True)
        """
        super(Control, self).__init__()

        # Instance variables
        self._name = name
        self._parent = parent

        self._controls = dict()
        self._register_with_window(self)

        # Geometry properties
        self._size = list(size)
        self._image = pygame.Surface(self._size, flags=pygame.SRCALPHA)
        self._rect = self.image.get_rect()
        self._rect.topleft = self.point_to_world(position)
        self._level = kwargs["level"] if "level" in kwargs else \
            (self._parent.level + 1 if isinstance(self._parent, Control) else 0)
        self._position = list(position)

        # UI propertise
        self._visible = kwargs["visible"] if "visible" in kwargs else True
        self._backcolor = kwargs["backcolor"] if "backcolor" in kwargs else self._parent.backcolor
        self._backimage = kwargs["backimage"] if "backimage" in kwargs else None
        self._borderstyle = kwargs["borderstyle"] if "borderstyle" in kwargs else BorderStyle.SOLID
        self._bordercolor = kwargs["bordercolor"] if "bordercolor" in kwargs else (0, 0, 0)
        self._borderwidth = kwargs["borderwidth"] if "borderwidth" in kwargs else 1
        self._canselect = kwargs["canselect"] if "canselect" in kwargs else False
        self._is_selected = kwargs["is_selected"] if "is_selected" in kwargs else False

        # Event properties
        self._keydown_pressed_keys = set()

    def __del__(self):
        self._unregister_with_window(self)

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
        # self._rect.topleft = self.position

        # If control is not visible, then none of its children are visible either.
        if self._visible:
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

    def handle_event(self, event):
        # Get mouse position relative to current control
        mouse_position = self.get_mouse_position()

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

    def point_to_control(self, world_point):
        if isinstance(self._parent, Window):
            return world_point
        world_parent_topleft = self._parent.point_to_world(self._parent.position)
        return [world_point[0] - world_parent_topleft[0], world_point[1] - world_parent_topleft[1]]

    def point_to_world(self, control_point):
        if isinstance(self._parent, Window):
            return control_point
        # return [self._parent.position[0] + control_point[0], self._parent.position[1] + control_point[1]]
        return self._parent.world_position[0] + control_point[0], self._parent.world_position[1] + control_point[1]

    def get_mouse_position(self):
        world_position = self._parent.get_mouse_position()
        return self.point_to_control(world_position)

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

    def move_by(self, dx, dy):
        self._rect.left += dx
        self._rect.top += dy
        for control in self._controls.values():
            control.move_by(dx, dy)

    def move_up_by(self, dy):
        dy = abs(dy)
        self.move_by(0, -dy)

    def move_down_by(self, dy):
        dy = abs(dy)
        self.move_by(0, dy)

    def move_left_by(self, dx):
        dx = abs(dx)
        self.move_by(-dx, 0)

    def move_right_by(self, dx):
        dx = abs(dx)
        self.move_by(dx, 0)

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
    def position(self):
        return self._position

    @position.setter
    def position(self, value):
        dx, dy = value[0] - self._position[0], value[1] - self._position[1]
        self._position = value
        self._rect.left += dx
        self._rect.top += dy

    @property
    def world_position(self):
        return self.rect.topleft

    @world_position.setter
    def world_position(self, value):
        raise NotImplementedError

    @property
    def size(self):
        return self._size

    @size.setter
    def size(self, value):
        raise NotImplementedError("TODO. Raise resize() event.")

    @property
    def left(self):
        return self.position[0]

    @left.setter
    def left(self, value):
        if value < 0:
            value = 0

        if value > self._parent.width - self.width:
            value = self._parent.width - self.width

        self._position[0] = value

    @property
    def top(self):
        return self.position[1]

    @top.setter
    def top(self, value):
        if value < 0:
            value = 0

        if value > self._parent.height - self.height:
            value = self._parent.height - self.height

        self._position[1] = value

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
    def visible(self):
        return self._visible

    @visible.setter
    def visible(self, value):
        raise NotImplementedError("TODO. Raise resize() event.")

    @property
    def backcolor(self):
        return self._backcolor

    @backcolor.setter
    def backcolor(self, value):
        raise NotImplementedError("TODO. Raise resize() event.")

    @property
    def backimage(self):
        return self._backimage

    @backimage.setter
    def backimage(self, value):
        raise NotImplementedError("TODO. Raise resize() event.")

    @property
    def borderstyle(self):
        return self._borderstyle

    @borderstyle.setter
    def borderstyle(self, value):
        raise NotImplementedError("TODO. Raise resize() event.")

    @property
    def borderwidth(self):
        return self._borderwidth

    @borderwidth.setter
    def borderwidth(self, value):
        raise NotImplementedError("TODO. Raise resize() event.")

    @property
    def bordercolor(self):
        return self._bordercolor

    @bordercolor.setter
    def bordercolor(self, value):
        raise NotImplementedError("TODO. Raise resize() event.")

    @property
    def canselect(self):
        return self._canselect

    @canselect.setter
    def canselect(self, value):
        raise NotImplementedError("TODO. Raise resize() event.")

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