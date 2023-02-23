import pygame
from typing import Union, List
from ggsolver.graph import *
from dataclasses import dataclass
from ggsolver.util import ColoredMsg


@dataclass
class LineStyle:
    """
    :param line_width: (int) Line width of borders.
    :param line_color: (int) (3-tuple of int) (R, G, B), each value between [0, 255]
    :param line_style: (str) Line style of borders.
        Currently, only "solid" is supported. Later, we will support "dashed".
    """
    line_width: int = 1
    line_style: str = "solid"
    line_color: tuple = (0, 0, 0)


class SM:
    def __init__(self, graph: Graph):
        self._graph = graph
        self._curr_state = None
        self._states_history = []
        self._action_history = []
        self._curr_step = 0
        self._len_history = float("inf")

    # TODO. Add methods for initialize, delta etc.
    #  Unclear whether inputs should be nodes or states?


DEFAULT_CELL_SIZE = (50, 50)
DEFAULT_BG_COLOR = (255, 255, 255)
DEFAULT_LINE_STYLE = LineStyle(line_width=3, line_style="solid", line_color=(0, 0, 0))
MOUSE_HOVER_ANIM_COLOR = (255, 222, 173)


class GWSim:
    def __init__(self, dim, sm,
                 window_size: Union[str, tuple[int, int]] = "auto",
                 cell_size: Union[str, tuple[int, int]] = "auto",
                 bg_color=DEFAULT_BG_COLOR,
                 mouse_hover_anim_color=MOUSE_HOVER_ANIM_COLOR,
                 grid_line_style=DEFAULT_LINE_STYLE,
                 mode=None,
                 fps=2,
                 show_help=True,
                 show_msg_box=False,
                 show_grid_lines=True,
                 show_mouse_hover_animation=True,
                 enable_sound=False,
                 caption="Gridworld demo"):
        """
        mode: ("manual", "auto", "hybrid")
        dim: (int > 0, int > 0)
        cell_size: (int >= 50, int >= 50)
        cell_size: (int >= 50 * max_x, int >= 50 * max_y)
        """

        # Instance variables
        self._dim = dim
        self._sm = sm
        self._mode = mode
        self._fps = fps
        self._show_help = show_help
        self._show_msg_box = show_msg_box
        self._show_grid_lines = show_grid_lines
        self._enable_sound = enable_sound
        self._bg_color = bg_color
        self._grid_line_style = grid_line_style
        self._window_caption = caption
        self._running = False
        self._paused = False

        # Initialize pygame window
        pygame.init()

        # Determine window and cell size
        self._window_size, self._cell_size = self._determine_window_cell_size(window_size, cell_size)

        # Set up pygame window
        pygame.display.set_caption(self._window_caption)
        self._screen = pygame.display.set_mode([self._window_size[0], self._window_size[1]])
        self._grid = pygame.Surface((self._cell_size[0] * self._dim[0], self._cell_size[1] * self._dim[1]))

        # Pygame events
        self._animate_mouse_hover = show_mouse_hover_animation
        self._mouse_hover_anim_color = mouse_hover_anim_color

        # Game objects
        self._game_objects = pygame.sprite.Group()
        self._bg_sprites = pygame.sprite.Group()
        self._p1_sprites = pygame.sprite.Group()
        self._p2_sprites = pygame.sprite.Group()
        self._p3_sprites = pygame.sprite.Group()

        # Initialization functions
        self._generate_bg_sprites()

    def allocate_sprite_rect(self, sprite, cell, is_background=False):
        """
        Returns a rectangle w.r.t. top-left of cell.
        """
        # If sprite is background sprite: it gets full allocation.
        if is_background:
            width = self._cell_size[0]
            height = self._cell_size[1]
            return 0, 0, width, height

        # If sprite is player character or NPC, then allocate smaller space within a cell.
        else:
            # Collect sprites in given cell
            sprites_in_cell = [
                spr for spr in self._p1_sprites.sprites() if tuple(spr.get_grid_coordinates()) == tuple(cell)
            ]
            # TODO. Add P2, P3 sprites as well.

            if len(sprites_in_cell) == 0:
                width = int(0.75 * self._cell_size[0])
                height = int(0.75 * self._cell_size[1])
                left = (self._cell_size[0] - width) // 2
                top = (self._cell_size[1] - height) // 2
                return left, top, width, height

            elif len(sprites_in_cell) == 1:
                idx = sprites_in_cell.index(sprite)

                width = int(0.40 * self._cell_size[0])
                height = int(0.40 * self._cell_size[1])

                if idx == 0:
                    left = int(0.05 * self._cell_size[0])
                    top = int(0.05 * self._cell_size[1])

                else:  # idx == 1:
                    left = int(0.5 * self._cell_size[0])
                    top = int(0.5 * self._cell_size[1])

                return left, top, width, height

    def add_p1_sprite(self, sprite):
        self._p1_sprites.add(sprite)

    def add_p2_sprite(self, sprite):
        self._p2_sprites.add(sprite)

    def add_p3_sprite(self, sprite):
        self._p3_sprites.add(sprite)

    def set_text(self, cell, text=None, style="Arial", size=20, text_color=(0, 0, 0), line_width=1, pos="topleft"):
        for spr in self._bg_sprites:
            if tuple(spr.get_grid_coordinates()) == tuple(cell):
                spr.set_text(text, style, size, text_color, line_width, pos)

    def clear_text(self, cell):
        self.set_text(cell)

    def toggle_grid_lines(self):
        self._show_grid_lines = not self._show_grid_lines
        for sprite in self._bg_sprites.sprites():
            sprite.toggle_grid_lines()

    def cell_size(self):
        return self._cell_size

    def run(self):
        clock = pygame.time.Clock()
        self._running = True
        while self._running:
            for event in pygame.event.get():
                self.event_handler(event)
            # self.update()
            # self.render(self._sm.curr_state)
            self.render(None)        # FIXME: FOR DEBUG

            # Set FPS
            clock.tick(self._fps)

    def event_handler(self, event):
        # Handle mouse_hover event.
        self.on_mouse_hover()

        if event.type == pygame.QUIT:
            self._running = False

        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_t:
                self.toggle_grid_lines()

        if event.type == pygame.MOUSEBUTTONDOWN:
            self.on_click()

    def update(self):
        # if self._show_grid_lines:
        #     self._bg_sprites.update(show_grid_lines=True)
        pass

    def render(self, state):
        # Screen background is black.
        # self._screen.fill((0, 0, 0))
        self._screen.fill((100, 200, 200))

        # Determine position of grid in window.
        grid_left = (self._screen.get_size()[0] - self._grid.get_size()[0]) // 2
        grid_top = (self._screen.get_size()[1] - self._grid.get_size()[1]) // 2

        # Update background sprites
        self._bg_sprites.update()
        self._bg_sprites.draw(self._grid)

        # Update player 1 sprites
        self._p1_sprites.update()
        self._p1_sprites.draw(self._grid)

        # Update player 2 sprites
        self._p2_sprites.update()
        self._p2_sprites.draw(self._grid)

        # Update player 3 (nature) sprites
        self._p3_sprites.update()
        self._p3_sprites.draw(self._grid)

        self._screen.blit(self._grid, (grid_left, grid_top))
        pygame.display.flip()

    def add_sprite(self, sprite):
        self._game_objects.add(sprite)

    def get_mouse_hover_animation_color(self):
        return self._mouse_hover_anim_color

    def on_mouse_hover(self):
        if self._animate_mouse_hover:
            mouse_pos = pygame.mouse.get_pos()
            for sprite in self._bg_sprites.sprites():
                sprite.on_mouse_hover(mouse_pos)

    def on_click(self):
        for spr in self._game_objects.sprites():
            spr.on_click(pygame.mouse.get_pos())

    def _generate_bg_sprites(self):
        x_max, y_max = self._dim
        for x in range(x_max):
            for y in range(y_max):
                cell_xy = BGSprite(parent=self, x=x, y=y, bg_color=self._bg_color,
                                   line_color=self._grid_line_style.line_color,
                                   line_width=self._grid_line_style.line_width,
                                   show_grid_lines=self._show_grid_lines)
                self._bg_sprites.add(cell_xy)
                # self._game_objects.add(cell_xy)

    def _auto_window_size(self):
        # Given gridworld dimensions, try 100 x 100 pix cells.
        # The window dimensions should not exceed screen size - 150 pix.
        # If not try 75 x 75 pix cells.
        # If not, raise error.
        # Fix window and cell sizes.
        # Handle given cell and/or window sizes.
        pass

    def _determine_window_cell_size(self, window_size, cell_size):
        display = pygame.display.Info()
        screen_size = [display.current_w - 100, display.current_h - 100]

        if window_size != "auto" and cell_size != "auto":
            # FIXME (hard coded): Grid should fit leaving at least 10 pixels border on all 4 sides of window.
            assert window_size[0] <= screen_size[0], \
                f"window width doesn't fit within screen width (available={screen_size[0]})."
            assert window_size[1] <= screen_size[1], \
                f"window width doesn't fit within screen width (available={screen_size[1]})."
            assert cell_size[0] >= 75 and cell_size[1] >= 75, "cell_size must be >= 75 x 75 pixels."
            assert cell_size[0] * self._dim[0] <= window_size[0] - 20, "cell_width * num_cols <= window_width"
            assert cell_size[1] * self._dim[1] <= window_size[1] - 20, "cell_height * num_rows <= window_height"

            print(ColoredMsg.ok(f"Setting user specified window_size: {window_size}, cell_size: {cell_size}"))
            return window_size, cell_size

        elif window_size != "auto" and cell_size == "auto":
            assert window_size[0] <= screen_size[0], \
                f"window width doesn't fit within screen width (available={screen_size[0]})."
            assert window_size[1] <= screen_size[1], \
                f"window width doesn't fit within screen width (available={screen_size[1]})."

            # Try cell size 100 px.
            if 100 * self._dim[0] <= window_size[0] - 20 and 100 * self._dim[1] <= window_size[1] - 20:
                print(ColoredMsg.ok(f"Setting user specified window_size: {window_size}, determined cell_size: {100}"))
                return window_size, (100, 100)

            # Try cell size 75 px.
            elif 75 * self._dim[0] <= window_size[0] - 20 and 75 * self._dim[1] <= window_size[1] - 20:
                print(ColoredMsg.ok(f"Setting user specified window_size: {window_size}, determined cell_size: {75}"))
                return window_size, (75, 75)

            # Currently, we will try only two cell sizes. Else give error.
            else:
                raise AssertionError("Neither 100 x 100 nor 75 x 75 cells could fit within given window size.")

        elif window_size == "auto" and cell_size != "auto":
            assert cell_size >= 75, "cell_size must be >= 75 x 75 pixels."
            assert cell_size[0] * self._dim[0] <= screen_size[0] - 20, "cell_width * num_cols <= screen_width"
            assert cell_size[1] * self._dim[1] <= screen_size[1] - 20, "cell_height * num_rows <= screen_height"

            window_width = cell_size[0] * self._dim[0] + 20
            window_height = cell_size[1] * self._dim[1] + 20

            print(ColoredMsg.ok(f"Setting determined window_size: {window_size}, user given cell_size: {cell_size}"))
            return (window_width, window_height), cell_size

        else:  # window_size == "auto" and cell_size == "auto":
            # Try cell size 100 px.
            #   Ensure window should leave 100 px border on screen, 20 px border between grid and window
            if 100 * self._dim[0] <= screen_size[0] - 120 and 100 * self._dim[1] <= screen_size[1] - 120:
                print(
                    ColoredMsg.ok(
                        f"Setting determined window_size: {(100 * self._dim[0] + 20, 100 * self._dim[1] + 20)}, "
                        f"cell_size: {100}"
                    )
                )
                return (100 * self._dim[0] + 20, 100 * self._dim[1] + 20), (100, 100)

            # Try cell size 75 px.
            elif 75 * self._dim[0] <= screen_size[0] - 120 and 75 * self._dim[1] <= screen_size[1] - 120:
                print(
                    ColoredMsg.ok(
                        f"Setting determined window_size: {(75 * self._dim[0] + 20, 75 * self._dim[1] + 20)}, "
                        f"cell_size: {75}"
                    )
                )
                return (75 * self._dim[0] + 20, 75 * self._dim[1] + 20), (75, 75)

            # Currently, we will try only two cell sizes. Else give error.
            else:
                raise AssertionError("Neither 100 x 100 nor 75 x 75 cells could fit within given window size.")


class GameObject(pygame.sprite.Sprite):
    def __init__(self, parent, x, y):
        """
        :param parent: (Player object)
        :param x: (int) X-coordinate.
        :param y: (int) Y-coordinate.
        """
        super(GameObject, self).__init__()

        # Store grid coordinates
        self._grid_coordinates = (x, y)

        # Set parent and child relation
        self._parent = parent
        self._parent.add_sprite(self)

        # Get surface allocation for image rectangle
        if isinstance(self, BGSprite):
            self._alloc_rect = self._parent.allocate_sprite_rect(sprite=self, cell=(x, y), is_background=True)
            print(f"BGSprite: {self._alloc_rect=}")
        else:
            self._alloc_rect = self._parent.allocate_sprite_rect(sprite=self, cell=(x, y))

        # Define surface to play with for the sprite.
        self.image = pygame.Surface(self._alloc_rect[2:])
        self.rect = self.image.get_rect()
        self.rect.topleft = [
            self._parent.cell_size()[0] * self._grid_coordinates[0] + self._alloc_rect[0],
            self._parent.cell_size()[1] * self._grid_coordinates[1] + self._alloc_rect[1]
        ]
        print(f"\t{self.rect.topleft=}")

    def get_grid_coordinates(self):
        return self._grid_coordinates

    def on_mouse_hover(self, mouse_pos):
        pass

    def on_click(self, mouse_pos):
        pass


class BGSprite(GameObject):
    def __init__(self, parent, x, y, bg_color=(255, 255, 255),
                 line_width=3, line_color=(0, 0, 0), show_grid_lines=True):
        """
        :param parent: (Player object)
        :param x: (int) X-coordinate.
        :param y: (int) Y-coordinate.
        :param bg_color: (3-tuple of int) (R, G, B), each value between [0, 255]
        :param line_width: (int) Line width of borders.
        :param line_color: (tuple[int, int, int]) Line color of borders.
        :param show_grid_lines: (bool) Show/hide grid line.
        """
        super(BGSprite, self).__init__(parent, x, y)
        self._bg_color = bg_color
        self._line_width = line_width
        self._line_color = line_color
        self._show_grid_lines = show_grid_lines
        self.image.fill(self._bg_color)

        self._text = None
        self._text_font = None
        self._text_color = (0, 0, 0)
        self._text_line_width = 1
        self._text_pos = "topleft"

    def update(self):
        if self._show_grid_lines:
            pygame.draw.rect(
                self.image,
                self._line_color,
                pygame.Rect(0, 0, self.rect.width, self.rect.height),
                self._line_width
            )
        else:
            self.image.fill(self._bg_color)

        # Handle text, if available
        if self._text is not None:
            textSurf = self._text_font.render(self._text, 1, (0, 0, 0))
            if self._text_pos == "topleft":
                self.image.blit(textSurf, [0, 0])
            else:
                raise ValueError("Currently, only top-left text position is allowed. Will add more later.")

    def toggle_grid_lines(self):
        self._show_grid_lines = not self._show_grid_lines

    def on_mouse_hover(self, mouse_pos):
        if self.rect.collidepoint(mouse_pos):
            self.image.fill(self._parent.get_mouse_hover_animation_color())
        else:
            self.image.fill(self._bg_color)

    def set_text(self, text, font_style="Arial", font_size=24, text_color=(0, 0, 0), line_width=1, pos="topleft"):
        self._text_font = pygame.font.SysFont(font_style, font_size)
        self._text = text
        self._text_color = text_color
        self._text_line_width = line_width
        self._text_pos = pos


class ObstacleSprite(GameObject):
    def __init__(self, parent, x, y, sprite_img):
        super(ObstacleSprite, self).__init__(parent, x, y)

        # Load image
        self._sprite_img = sprite_img
        self._sprite_img_surf = pygame.image.load(self._sprite_img).convert_alpha()
        pygame.transform.scale(self._sprite_img_surf, self.rect.size, self.image)


class PlayerSprite(GameObject):
    def __init__(self, parent, x, y, sprite_img, line_color=(0, 255, 0)):
        super(PlayerSprite, self).__init__(parent, x, y)

        # Load image
        self._sprite_img = sprite_img
        self._sprite_img_surf = pygame.image.load(self._sprite_img).convert_alpha()
        pygame.transform.scale(self._sprite_img_surf, self.rect.size, self.image)

        # Selection state
        self._is_selected = False
        self._line_color = line_color
        self._line_width = 4

    def on_click(self, mouse_pos):
        if self.rect.collidepoint(mouse_pos):
            self._is_selected = not self._is_selected

    def update(self):
        if self._is_selected:
            # When selected, draw border
            pygame.draw.rect(
                self.image,
                self._line_color,
                pygame.Rect(0, 0, self.rect.width, self.rect.height),
                self._line_width
            )

            # Handle any connected tasks to be handled on select.
            self.on_select()

        else:
            # When unselected, remove border
            self._sprite_img_surf = pygame.image.load(self._sprite_img).convert_alpha()
            pygame.transform.scale(self._sprite_img_surf, self.rect.size, self.image)

            # Handle any connected tasks to be handled on select.
            self.on_unselect()

    def on_select(self):
        pass

    def on_unselect(self):
        pass


class UpArrow(GameObject):
    def __init__(self, parent, x, y):
        super(UpArrow, self).__init__(parent, x, y)
        

if __name__ == '__main__':
    sim = GWSim(
        dim=(5, 5),
        sm=None,
        # window_size=(700, 700),
        # cell_size=(75, 75),
        show_grid_lines=True,
        fps=60,
        grid_line_style=LineStyle(1, "solid", (0, 0, 0))
    )
    # for spr in sim._bg_sprites.sprites():
    #     print(spr, spr.rect)
    player = PlayerSprite(sim, 2, 1, "0.png")
    obs = PlayerSprite(sim, 2, 1, "0.png")

    sim.add_p1_sprite(player)
    sim.add_p3_sprite(obs)
    sim.set_text((0, 0), "Hello!")
    sim.run()
