"""
Programmer's notes:


"""
import json

import numpy as np
import pygame
import os
import pathlib
import itertools
import random

import ggsolver.mdp as mdp
import ggsolver.gridworld as gw
import ggsolver.gridworld.util as gw_utils
from collections import namedtuple

curr_file_path = pathlib.Path(__file__).parent.resolve()


class BankHeistWindow(gw.Window):
    def __init__(self, name, size, game_config, **kwargs):
        super(BankHeistWindow, self).__init__(name, size, **kwargs)
        with open(game_config, "r") as file:
            self._game_config = json.load(file)

        if self._game_config["game"] != "Bank Heist":
            raise ValueError("The game is not Bank Heist.")

        # Construct grid
        self._terrain = np.array(self._game_config["terrain"])
        grid_size = tuple(reversed(self._terrain.shape))
        self.grid = gw.Grid(
            name="grid1",
            parent=self,
            position=(0, 0),
            size=size,
            grid_size=grid_size,
            backcolor=gw.COLOR_BEIGE,
            dockstyle=gw.DockStyle.TOP_LEFT,
            on_cell_leave=self.on_cell_leave,
            on_cell_enter=self.on_cell_enter,
        )

        for x in range(grid_size[0]):
            for y in range(grid_size[1]):
                if self._terrain[y, x] == 0:
                    self.grid[x, y].backcolor = gw.COLOR_GRAY51

        # Create character
        self._robber = Character(
            name="robber",
            parent=self.grid[0, 7],
            position=(0, 0),
            size=(0.75 * self.grid[0, 0].width, 0.75 * self.grid[0, 0].height),
            dockstyle=gw.DockStyle.CENTER,
            sprites=self._game_config["p1"]["sprites"],
            backcolor=gw.COLOR_TRANSPARENT,
            init_sprite="N",
        )

        # Police cars
        # PATCH: Hard coded to two for now.
        self._police1 = Character(
            name="police1",
            parent=self.grid[3, 1],
            position=(0, 0),
            size=(0.75 * self.grid[0, 0].width, 0.75 * self.grid[0, 0].height),
            dockstyle=gw.DockStyle.CENTER,
            sprites=self._game_config["p2"]["sprites"],
            backcolor=gw.COLOR_TRANSPARENT,
            # visible=False,
            init_sprite="N",
        )
        self._police2 = Character(
            name="police2",
            parent=self.grid[4, 1],
            position=(0, 0),
            size=(0.75 * self.grid[0, 0].width, 0.75 * self.grid[0, 0].height),
            dockstyle=gw.DockStyle.CENTER,
            sprites=self._game_config["p2"]["sprites"],
            backcolor=gw.COLOR_TRANSPARENT,
            # visible=False,
            init_sprite="N",
        )

        # Banks
        bank1_pos = self._game_config["banks"]["banks.1"]
        bank2_pos = self._game_config["banks"]["banks.2"]
        self._bank1 = Character(
            name="bank1",
            parent=self.grid[bank1_pos[0], bank1_pos[1]],
            position=(0, 0),
            size=(0.75 * self.grid[0, 0].width, 0.75 * self.grid[0, 0].height),
            dockstyle=gw.DockStyle.CENTER,
            sprites=self._game_config["banks"]["sprites"],
            backcolor=gw.COLOR_TRANSPARENT,
            visible=True,
            init_sprite="front",
        )
        self._bank2 = Character(
            name="bank2",
            parent=self.grid[bank2_pos[0], bank2_pos[1]],
            position=(0, 0),
            size=(0.75 * self.grid[0, 0].width, 0.75 * self.grid[0, 0].height),
            dockstyle=gw.DockStyle.CENTER,
            sprites=self._game_config["banks"]["sprites"],
            backcolor=gw.COLOR_TRANSPARENT,
            visible=True,
            init_sprite="front",
        )

        # Gas station
        gas1_pos = self._game_config["gas"]["gas.1"]
        self._gas = Character(
            name="gas",
            parent=self.grid[gas1_pos[0], gas1_pos[1]],
            position=(0, 0),
            size=(0.75 * self.grid[0, 0].width, 0.75 * self.grid[0, 0].height),
            dockstyle=gw.DockStyle.CENTER,
            sprites=self._game_config["gas"]["sprites"],
            backcolor=gw.COLOR_TRANSPARENT,
            visible=True,
            init_sprite="front",
        )

    def sm_update(self, sender, event_args):
        print(f"sm_update: {event_args}")

    def on_cell_leave(self, sender, event_args):
        # print(f"on_cell_leave: {sender.name=}, {event_args=}")
        if sender.name == event_args.trigger.name:
            self.arrange_controls_in_cell(event_args.trigger)

    def on_cell_enter(self, sender, event_args):
        # print(f"on_cell_enter: {sender.name=}, {event_args=}")
        if sender.name == event_args.trigger.name:
            self.arrange_controls_in_cell(event_args.trigger)

    def arrange_controls_in_cell(self, cell):
        if len(cell.controls) == 0:
            pass

        elif len(cell.controls) == 1:
            print(f"arrange_controls_in_cell: {cell.name}, {len(cell.controls)=}")
            control = list(cell.controls.values())[0]
            control.width = 0.75 * cell.width
            control.height = 0.75 * cell.height
            control.dock = gw.DockStyle.CENTER

        elif len(cell.controls) == 2:
            print(f"arrange_controls_in_cell: {cell.name}, {len(cell.controls)=}")

            control0 = list(cell.controls.values())[0]
            control0.dock = gw.DockStyle.TOP_LEFT
            control0.width = 0.5 * cell.width
            control0.height = 0.5 * cell.height

            control1 = list(cell.controls.values())[1]
            control1.dock = gw.DockStyle.BOTTOM_RIGHT
            control1.width = 0.5 * cell.width
            control1.height = 0.5 * cell.height
        else:
            print(f"Not supported")


class Character(gw.Control):
    def __init__(self, name, parent, position, size, **kwargs):
        """
        Additional kwargs:
        * sprites: (Dict[str, PathLike]) Mapping of sprite name to sprite image.
        * init_sprite: (str) Name of initial sprite to use.
        """
        super(Character, self).__init__(name, parent, position, size, on_key_down=self.on_key_down, **kwargs)

        self._sprite_files = kwargs["sprites"] if "sprites" in kwargs else dict()
        self._sprites = {name: None for name, file in self._sprite_files.items()}
        # self._curr_sprite = None
        # PATCH
        self._curr_sprite = kwargs["init_sprite"] if "init_sprite" in kwargs else None

        self.add_event_handler(pygame.MOUSEBUTTONDOWN, self._on_select_changed)

    def __repr__(self):
        return f"<{self.__class__.__name__} at {self.parent.name}>"

    def _on_select_changed(self, sender, event_args):
        self._is_selected = not self._is_selected
    
    def update(self):
        super(Character, self).update()

        if self._curr_sprite is not None:
            if self._sprites[self._curr_sprite] is None:
                self._sprites[self._curr_sprite] = pygame.image.load(os.path.join(curr_file_path, pathlib.Path(self._sprite_files[self._curr_sprite])))
                self._sprites[self._curr_sprite] = pygame.transform.scale(self._sprites[self._curr_sprite], (50, 50))
            self._backimage = self._sprites[self._curr_sprite]

    def on_key_down(self, sender, event_args):
        if self.name == "robber":
            if event_args.key == pygame.K_RIGHT:
                self._curr_sprite = "E"
                (x, y) = self.parent.name
                if (x + 1, y) in self.parent.parent.controls:
                    self.parent = self.parent.parent[x + 1, y]

            if event_args.key == pygame.K_LEFT:
                self._curr_sprite = "W"
                (x, y) = self.parent.name
                if (x - 1, y) in self.parent.parent.controls:
                    self.parent = self.parent.parent[x - 1, y]

            if event_args.key == pygame.K_UP:
                self._curr_sprite = "N"
                (x, y) = self.parent.name
                if (x, y + 1) in self.parent.parent.controls:
                    self.parent = self.parent.parent[x, y + 1]

            if event_args.key == pygame.K_DOWN:
                self._curr_sprite = "S"
                (x, y) = self.parent.name
                if (x, y - 1) in self.parent.parent.controls:
                    self.parent = self.parent.parent[x, y - 1]

        if event_args.key == pygame.K_h:
            self.visible = not self.visible


class BankHeistMDP(mdp.QualitativeMDP):
    def __init__(self, game_config):
        super(BankHeistMDP, self).__init__()
        with open(game_config, "r") as file:
            self._game_config = json.load(file)

        self._terrain = self._orient_terrain(np.array(self._game_config["terrain"]))
        self._p2_1_accessible = self._orient_terrain(np.array(self._game_config["p2"]["p2.1"]["accessible region"]))
        self._p2_2_accessible = self._orient_terrain(np.array(self._game_config["p2"]["p2.2"]["accessible region"]))
        self._grid_size = self._terrain.shape

    def states(self):
        """
        State representation: (p1.cell, p2.1.cell, p2.2.cell, p1.gas)
        :return:
        """
        x_max, y_max = self._grid_size
        p1_walkable_cells = [(x, y) for x in range(x_max) for y in range(y_max) if self._terrain[x, y] == 1]
        p1_gas = self._game_config["p1"]["gas capacity"]
        p2_1_walkable_cells = [(x, y) for x in range(x_max) for y in range(y_max) if self._p2_1_accessible[x, y] == 1]
        p2_2_walkable_cells = [(x, y) for x in range(x_max) for y in range(y_max) if self._p2_2_accessible[x, y] == 1]

        return list(
            filter(self._is_state_valid,
                   itertools.product(p1_walkable_cells, p2_1_walkable_cells, p2_2_walkable_cells, range(p1_gas)))
        )

    def actions(self):
        return [
            gw_utils.GW_ACT_N,
            gw_utils.GW_ACT_S,
            gw_utils.GW_ACT_E,
            gw_utils.GW_ACT_W,
        ]

    def delta(self, state, act):
        # Decouple state
        p1_cell, p2_1_cell, p2_2_cell, p1_gas = state

        # Base case
        if p1_gas == 0:
            return [state]

        # Generate all possible next states
        next_states = set()
        next_p1_cell = gw_utils.move(p1_cell, act)
        for act1, act2 in itertools.product(self.actions(), self.actions()):
            next_p2_1_cell = gw_utils.move(p2_1_cell, act1)
            next_p2_2_cell = gw_utils.move(p2_2_cell, act2)
            next_states.add((next_p1_cell, next_p2_1_cell, next_p2_2_cell, p1_gas - 1))

        # Filter unacceptable states
        filter_states = set()
        for next_state in next_states:
            next_p1_cell, next_p2_1_cell, next_p2_2_cell, _ = next_state

            if self._terrain[next_p1_cell[0], next_p1_cell[1]] == 0 or \
                    self._terrain[next_p2_1_cell[0], next_p2_1_cell[1]] == 0 or \
                    self._terrain[next_p2_2_cell[0], next_p2_2_cell[1]] == 0:
                filter_states.add(next_state)

        # Return
        return list(next_states - filter_states)

    def _is_state_valid(self, state):
        p1_cell, p2_1_cell, p2_2_cell, p1_gas = state

        if self._terrain[p1_cell[0], p1_cell[1]] == 0 or \
                self._p2_1_accessible[p2_1_cell[0], p2_1_cell[1]] == 0 or \
                self._p2_2_accessible[p2_2_cell[0], p2_2_cell[1]] == 0 or \
                self._game_config["p1"]["gas capacity"] <= p1_gas:
            return False

        return True

    def _matrix_flip(self, mat, axis):
        if not hasattr(mat, 'ndim'):
            mat = np.asarray(mat)
        indexer = [slice(None)] * mat.ndim
        try:
            indexer[axis] = slice(None, None, -1)
        except IndexError:
            raise ValueError("axis =% i is invalid for the % i-dimensional input array"
                             % (axis, mat.ndim))
        return mat[tuple(indexer)]

    def _orient_terrain(self, mat):
        return self._matrix_flip(np.transpose(mat), axis=0)


if __name__ == '__main__':
    conf = os.path.join(curr_file_path, "saved_games", "game_2022_11_22_23_29.conf")
    print(f"Using configuration file: {conf=}")

    game = BankHeistMDP(game_config=conf)
    arbitrary_state = random.choice(game.states())
    print("Executing: game = BankHeistMDP(game_config=conf)")
    print(f"Executing: random.choice(game.states())={arbitrary_state}")

    game.initialize(arbitrary_state)
    graph = game.graphify(pointed=True)
    print("Executing: graph = game.graphify(pointed=True)")


    # window = BankHeistWindow(name="Bank Heist", size=(660, 480), game_config=conf)
    # window.run()
