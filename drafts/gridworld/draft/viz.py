import random
import sys

import pygame
import util
from multiprocessing import Process


"""
Workflow. 

1. Create GWSim instance. 
2. (Mandatory). Define god player's class.
3. (Mandatory). Define god player's window.
4. (Optional). Define other players' class. This will define their perspectives. 
5. (Optional). Attach other players' windows. This will connect their state perceptions to gridworld. 
6. (Optional). Attach any plugins. 
"""


class SM:
    def __init__(self):
        self._state = None
        self._states_history = []
        self._action_history = []
        self._curr_step = 0
        self._len_history = float("inf")

    def initialize(self, state):
        pass

    def update(self, state, inp):
        """
        Consumes state, input to produce next state and output.
        """
        pass


class GWSim:
    STANDALONE = "standalone"
    SERVER = "server"

    LAYOUT_CLASSIC = "classic"

    def __init__(self,
                 gw_size,
                 mode=STANDALONE,
                 graph=None,
                 server=None):
        """
        mode: (str) decides whether to connect to a server or to load a game graph.
        graph: (Graph or None). The graph used to run the simulation.
        server: (str) IP address of the server with port (e.g., `0.0.0.0:8080`).
        """
        # Mode
        self._mode = mode
        self._gw_size = gw_size
        self._graph = None
        self._server = None

        # Players and Game objects
        self._players = {
            "god": None,
            "p1": None,
            "p2": None,
            "env": None,
        }
        self._game_objects = pygame.sprite.Group()      # (Sprite group containing all sprites)

        # Plugins (e.g., message windows etc.)
        self._plugins = dict()

        # Pygame window(s)
        self._windows = dict()
        self._layout = self.LAYOUT_CLASSIC

        # Status flags
        self._is_initialized = False

        # Initialize the simulation
        if mode == self.STANDALONE:
            self.init_with_graph(graph)
        elif mode == self.SERVER:
            self.init_with_server(server)
        else:
            raise ValueError(f"Input mode must be either {self.STANDALONE} or {self.SERVER}.")

    def set_player(self, name, player):
        assert isinstance(player, Player)
        assert name in self._players, "Input name should be one of {god, p1, p2, env}."
        self._players[name] = player
        for obj in player.game_objects():
            self._game_objects.add(obj)
        player.set_parent(self)

    def rem_player(self, name):
        if name in self._players:
            for obj in self._players[name].game_objects():
                self._game_objects.remove(obj)
        self._players.pop(name)

    def init_windows(self):
        pass

    def init_with_graph(self, graph):
        # TODO. Check important graph properties required for simulation.
        self._graph = graph

    def init_with_server(self, server):
        # TODO. Initialize connection to server.
        pass

    def init_plugins(self):
        pass

    def set_layout(self, layout=LAYOUT_CLASSIC):
        self._layout = layout

    def is_initialized(self):
        return self._is_initialized

    def mode(self):
        return self._mode

    def windows(self):
        return self._windows

    def run(self):
        # Initialize pygame windows, plugins and generate layout
        self.init_windows()
        self.init_plugins()
        self.set_layout(self._layout)

        if len(self.windows()) == 0:
            print("[ERROR] No windows configured. Running in NOVIS mode.")

        pygame_processes = dict()
        for p_name, player in self._players.items():
            if player is not None:
                process = Process(target=player.run)
                process.daemon = True
                pygame_processes[p_name] = process

        for p_name, process in pygame_processes.items():
            process.start()
            print(f"[INFO] Started process for {p_name}.")

        while True:
            print({p_name: process.is_alive() for p_name, process in pygame_processes.items()}, end="\r")
            if not any(process.is_alive() for p_name, process in pygame_processes.items()):
                break

        print()
        print("Exited.")

    def gw_size(self):
        return self._gw_size


class GWWindow(SM):
    def __init__(self, parent, window_size):
        super(GWWindow, self).__init__()

        # Store reference to parent
        self._parent = parent

        # Window parameters
        self._window_size = window_size
        self._window = None
        self._visible = False

        # Game objects
        self._game_objects = pygame.sprite.Group()
        self._bg_sprites = pygame.sprite.Group()

        # Initialization functions
        self._generate_bg_sprites()

    def cell_size(self):
        x_max, y_max = self.parent().gw_size()
        width, height = self.window_size()
        return width // x_max, height // y_max

    def window_size(self):
        return self._window_size

    def parent(self):
        return self._parent

    def set_visible(self, value):
        raise NotImplementedError

    def set_parent(self, value):
        assert isinstance(value, GWSim)
        self._parent = value

    def run(self):
        self._window = pygame.display.set_mode(self.window_size())
        # self._window.fill(self.bgcolor)
        while True:
            # Handle events
            events = pygame.event.get()
            for e in events:
                if e.type == pygame.QUIT:
                    sys.exit(0)
                if e.type == pygame.KEYDOWN:
                    if e.key == pygame.K_ESCAPE:
                        sys.exit(0)  # close this specific process

            # Draw sprites
            self._bg_sprites.update()
            self._bg_sprites.draw(self._window)

            # Update display
            pygame.display.update()

    def _generate_bg_sprites(self):
        x_max, y_max = self._parent.gw_size()
        for x in range(x_max):
            for y in range(y_max):
                cell_xy = Cell(parent=self, x=x, y=y)
                self._bg_sprites.add(cell_xy)
                self._game_objects.add(cell_xy)


class Player(SM):
    def __init__(self,
                 name,
                 perceptual_graph):

        # Initialize player state machine
        super(Player, self).__init__()

        # Player's game
        self._graph = perceptual_graph      # Perceptual game graph

        # Thought bubble visualization
        self._parent = None                 # GWSim instance.
        self._window = None                 # Pygame window
        self._tb_visible = False            # Thought bubble window visibility.

        self.size = (200, 200)
        self.screen = None
        self.bgcolor = (random.randint(0, 255), 0, 0)
        self.name = name

    def set_tb_visibility(self, value):
        self._tb_visible = value
        self._parent.set_layout()

    def set_parent(self, value):
        assert isinstance(value, GWSim)
        self._parent = value

    def game_objects(self):
        # TODO. Implement.
        return []

    def perspective(self):
        return None

    def run(self):
        self.screen = pygame.display.set_mode(self.size)
        self.screen.fill(self.bgcolor)
        while True:
            events = pygame.event.get()
            for e in events:
                if e.type == pygame.QUIT:
                    sys.exit(0)
                if e.type == pygame.KEYDOWN:
                    if e.key == pygame.K_ESCAPE:
                        sys.exit(0)  # close this specific process

            pygame.display.update()


class GameObject(pygame.sprite.Sprite):
    pass


class Cell(GameObject):
    def __init__(self, parent, x, y,
                 bg_img=None, bg_color=(255, 255, 255),
                 line_width=1, line_style="solid", line_color=(0, 0, 0)):
        """
        :param parent: (Player object)
        :param x: (int) X-coordinate.
        :param y: (int) Y-coordinate.
        :param bg_color: (3-tuple of int) (R, G, B), each value between [0, 255]
        :param bg_img: (TBD) Background image.
        :param line_width: (int) Line width of borders.
        :param line_color: (int) (3-tuple of int) (R, G, B), each value between [0, 255]
        :param line_style: (str) Line style of borders.
            Currently, only "solid" is supported. Later, we will support "dashed".
        """
        super(Cell, self).__init__()
        self._parent = parent
        self._pos = (x, y)
        self._bg_img = bg_img
        self._bg_color = bg_color
        self._line_width = line_width
        self._line_color = line_color
        self._line_style = line_style

        self._image = pygame.Surface(self._parent.cell_size())
        if bg_img is None:
            self._image.fill(self._bg_color)
            self._image.set_colorkey(self._bg_color)
        else:
            raise NotImplementedError("Background images are not supported in this version. ")

        pygame.draw.rect(self.image,
                         self._line_color,
                         pygame.Rect(0, 0, self._image.get_width(), self._image.get_height()))

        self.rect = self.image.get_rect()


if __name__ == '__main__':

    sim = GWSim(gw_size=(4, 4))

    # Add player
    p1 = Player("player 1", None)
    sim.set_player("p1", p1)

    # Add perspective of player.
    w1 = GWWindow(sim, window_size=(400, 400))

    # p2 = Player("player 2", None)
    # sim.set_player("p2", p2)

    sim.run()
