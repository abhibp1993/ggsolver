import pygame


class SM:
    def __init__(self):
        self._state = None
        self._states_history = []
        self._action_history = []
        self._curr_step = 0
        self._len_history = float("inf")

    def initialize(self, state):
        pass

    def delta(self, state, inp):
        """
        Consumes state, input to produce next state and output.
        """
        pass


class GWSim:
    def __init__(self, gw_size):
        self._gw_size = gw_size
        self._players = dict()
        self._windows = dict()
        self._plugins = dict()

    def get_player(self, name):
        pass

    def set_player(self, name, player):
        pass

    def rem_player(self, name):
        pass

    def attach_perspective(self, name, window):
        pass

    def run(self):
        pass


class GWWindow:
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


class GameObject(pygame.sprite.Sprite):
    pass


class Cell(GameObject):
    pass


class GWPlayer(SM):
    pass