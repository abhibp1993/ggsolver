import inspect
import itertools

import ggsolver.util as util
from ggsolver.gridworld.models2 import *
from ggsolver.models import Game


class TicTacToe(Game):
    def __init__(self, init_player=1):
        super(TicTacToe, self).__init__(is_deterministic=True)
        self.init_player = init_player
        self.winning_cell_combinations = [
            (0, 3, 6),
            (1, 4, 7),
            (2, 5, 8),
            (0, 1, 2),
            (3, 4, 5),
            (6, 7, 8),
            (0, 4, 8),
            (2, 4, 6),
        ]

    def states(self):
        # player 0: cell is unmarked, 1 means marked by P1, 2 means marked by P2.
        mark = [0, 1, 2]
        return list(itertools.product(mark, repeat=9))

    def actions(self):
        # (cell, player) means mark 'cell' to be 'player's cell.
        return [(cell, player) for cell in range(9) for player in range(3)]

    def delta(self, state, act):
        cell, player = act
        turn = self.turn(state)
        if state[cell] == 0 and turn == player:
            n_state = list(state)
            n_state[cell] = player
            return tuple(n_state)
        return state

    def atoms(self):
        return ["win", "draw", "lose"]

    def label(self, state):
        for c1, c2, c3 in self.winning_cell_combinations:
            if state[c1] == state[c2] == state[c3] == 1:
                return ["win"]
            if state[c1] == state[c2] == state[c3] == 2:
                return ["lose"]
        if all(cell_marked_with in [1, 2] for cell_marked_with in state):
            return ["draw"]
        return []

    def turn(self, state):
        # if state.count(1) % 2 == 0:
        if state.count(1) == state.count(2):
            return self.init_player
        else:
            return 2 if self.init_player == 1 else 1

    def init_state(self):
        return (0, ) * 9


class TicTacToeSim(GWSim):
    def __init__(self, size, init_player=1, **kwargs):
        # Define game and its graph
        self._game = TicTacToe(init_player=init_player)
        self._game.initialize(state=self._game.init_state())
        self._graph = self._game.graphify(pointed=True)

        # Initialize GWSim
        super(TicTacToeSim, self).__init__(name="TicTacToe", size=size, graph=self._graph, **kwargs)

        # PATCH
        self._curr_state = self._game.init_state()

        # Add controls
        self.grid = Grid(name="grid", parent=self, position=(0, 0), size=size, grid_size=(3, 3), cls_cell=TicTacToeCell)
        self.add_control(self.grid)

        # Register on_click event
        self.grid[0, 0].on_mouse_click = self.mouse_click_on_0
        self.grid[1, 0].on_mouse_click = self.mouse_click_on_1
        self.grid[2, 0].on_mouse_click = self.mouse_click_on_2
        self.grid[0, 1].on_mouse_click = self.mouse_click_on_3
        self.grid[1, 1].on_mouse_click = self.mouse_click_on_4
        self.grid[2, 1].on_mouse_click = self.mouse_click_on_5
        self.grid[0, 2].on_mouse_click = self.mouse_click_on_6
        self.grid[1, 2].on_mouse_click = self.mouse_click_on_7
        self.grid[2, 2].on_mouse_click = self.mouse_click_on_8

    def step(self, act):
        curr_state = self._curr_state
        next_state = self._game.delta(curr_state, act)
        if next_state != curr_state:
            self._curr_state = next_state
            if act[1] == self._game.init_player:
                self.grid[self.cell2grid(act[0])].show_picture(TicTacToeCell.PICTURE_X)
            else:
                self.grid[self.cell2grid(act[0])].show_picture(TicTacToeCell.PICTURE_O)
        print(f"{self._curr_state=}")

    def mouse_click_on_0(self, event_args):
        print(f"Call 0: {event_args}")
        player = self._game.turn(self._curr_state)
        self.step((0,  player))

    def mouse_click_on_1(self, event_args):
        print(f"Call 1: {event_args}")
        player = self._game.turn(self._curr_state)
        self.step((1,  player))

    def mouse_click_on_2(self, event_args):
        print(f"Call 2: {event_args}")
        player = self._game.turn(self._curr_state)
        self.step((2,  player))

    def mouse_click_on_3(self, event_args):
        print(f"Call 3: {event_args}")
        player = self._game.turn(self._curr_state)
        self.step((3,  player))

    def mouse_click_on_4(self, event_args):
        print(f"Call 4: {event_args}")
        player = self._game.turn(self._curr_state)
        self.step((4,  player))

    def mouse_click_on_5(self, event_args):
        print(f"Call 5: {event_args}")
        player = self._game.turn(self._curr_state)
        self.step((5,  player))

    def mouse_click_on_6(self, event_args):
        print(f"Call 6: {event_args}")
        player = self._game.turn(self._curr_state)
        self.step((6,  player))

    def mouse_click_on_7(self, event_args):
        print(f"Call 7: {event_args}")
        player = self._game.turn(self._curr_state)
        self.step((7,  player))

    def mouse_click_on_8(self, event_args):
        print(f"Call 8: {event_args}")
        player = self._game.turn(self._curr_state)
        self.step((8,  player))

    def cell2grid(self, cellid):
        q, r = divmod(cellid, 3)
        return r, q


class TicTacToeCell(Cell):
    PICTURE_X = "sprites/x.jpg"
    PICTURE_O = "sprites/o.png"

    def __init__(self, name, parent, position, size, **kwargs):
        super(TicTacToeCell, self).__init__(name, parent, position, size, **kwargs)
        self._picture = None

    def show_picture(self, picture_path):
        if picture_path is None:
            self._picture = None

        elif picture_path == TicTacToeCell.PICTURE_X:
            self._picture = pygame.image.load(TicTacToeCell.PICTURE_X).convert_alpha()
            self._picture = pygame.transform.scale(self._picture, (int(0.9 * self.width), int(0.9 * self.height)))

        elif picture_path == TicTacToeCell.PICTURE_O:
            self._picture = pygame.image.load(TicTacToeCell.PICTURE_O).convert_alpha()
            self._picture = pygame.transform.scale(self._picture, (int(0.9 * self.width), int(0.9 * self.height)))

        else:
            print(util.ColoredMsg.warn(f"[WARN] Picture for X/O could not be loaded."))

    def update(self):
        # print(f"Called: {self}.{inspect.stack()[0][3]}")
        super(TicTacToeCell, self).update()
        if self._picture is not None:
            self.image.blit(self._picture, (int(0.05 * self.width), int(0.05 * self.height)))


def check_game():
    game = TicTacToe()
    game.initialize(game.init_state())
    graph = game.graphify(pointed=True)
    print(f"{graph.number_of_nodes()}")
    print(f"{graph.number_of_edges()}")


if __name__ == '__main__':
    sim = TicTacToeSim(size=(300, 300), backcolor=(255, 222, 200))
    sim.run()
