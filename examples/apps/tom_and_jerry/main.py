"""
Programmer's notes:
    - Input grid coordinate is XY.
    - TomJerryGame terrain is XY.
    - Only in rendering we will use Row-Column to represent gridworld.
"""
import json

import numpy as np
import pygame
import os
import pathlib
import itertools
import random

import ggsolver.gridworld as gw
import ggsolver.gridworld.util as gw_utils
import ggsolver.decoy_alloc.cleanup.models as decoy_models
import ggsolver.decoy_alloc.cleanup.solvers as solvers
import ggsolver.dtptb.reach as reach
import ggsolver.graph as gg_graph

from ggsolver import dtptb

curr_file_path = pathlib.Path(__file__).parent.resolve()


class TomJerryWindow(gw.Window):
    def __init__(self, name, size, game_config, **kwargs):
        super(TomJerryWindow, self).__init__(name, size, **kwargs)
        with open(game_config, "r") as file:
            self._game_config = json.load(file)

        if self._game_config["game"] != "Tom and Jerry":
            raise ValueError("The game is not Tom and Jerry.")

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
                if self._terrain[grid_size[1] - 1 - y, x] == 0:
                    self.grid[x, y].backcolor = gw.COLOR_GRAY51
                if self._terrain[grid_size[1] - 1 - y, x] == 2:
                    self.grid[x, y].backcolor = gw.COLOR_ROSYBROWN

        # Create character
        self._jerry = Character(
            name="jerry",
            parent=self.grid[1, 2],
            position=(0, 0),
            size=(0.75 * self.grid[0, 0].width, 0.75 * self.grid[0, 0].height),
            dockstyle=gw.DockStyle.CENTER,
            sprites=self._game_config["jerry"]["sprites"],
            backcolor=gw.COLOR_TRANSPARENT,
            init_sprite="N",
        )

        # Create cat (tom)
        self._cat = Character(
            name="cat",
            parent=self.grid[1, 6],
            position=(0, 0),
            size=(0.75 * self.grid[0, 0].width, 0.75 * self.grid[0, 0].height),
            dockstyle=gw.DockStyle.CENTER,
            sprites=self._game_config["tom"]["sprites"],
            backcolor=gw.COLOR_TRANSPARENT,
            # visible=False,
            init_sprite="N",
        )

        # Create cheese
        cheese1_pos = self._game_config["cheese"]["cheese.1"]

        self._cheese1 = Character(
            name="cheese1",
            parent=self.grid[cheese1_pos[0], cheese1_pos[1]],
            position=(0, 0),
            size=(0.75 * self.grid[0, 0].width, 0.75 * self.grid[0, 0].height),
            dockstyle=gw.DockStyle.CENTER,
            sprites=self._game_config["cheese"]["sprites"],
            backcolor=gw.COLOR_TRANSPARENT,
            visible=True,
            init_sprite="front",
        )

    def add_trap(self, x, y):
        self._trap = Character(
            name="trap",
            parent=self.grid[x, y],
            position=(x, y),
            size=(0.75 * self.grid[0, 0].width, 0.75 * self.grid[0, 0].height),
            dockstyle=gw.DockStyle.CENTER,
            sprites=self._game_config["trap"]["sprites"],
            backcolor=gw.COLOR_TRANSPARENT,
            visible=True,
            init_sprite="front",
        )

    def add_fake(self, x, y):
        self._fake = Character(
            name="fake",
            parent=self.grid[x, y],
            position=(x, y),
            size=(0.75 * self.grid[0, 0].width, 0.75 * self.grid[0, 0].height),
            dockstyle=gw.DockStyle.CENTER,
            sprites=self._game_config["fake_target"]["sprites"],
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
                self._sprites[self._curr_sprite] = pygame.image.load(
                    os.path.join(curr_file_path, pathlib.Path(self._sprite_files[self._curr_sprite])))
                self._sprites[self._curr_sprite] = pygame.transform.scale(self._sprites[self._curr_sprite], (50, 50))
            self._backimage = self._sprites[self._curr_sprite]

    def on_key_down(self, sender, event_args):
        if self.name == "jerry":
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


class Terrain:
    def __init__(self, terrain_array: np.ndarray):
        self.terrain = terrain_array
        self._y_max, self._x_max = self.terrain.shape

    def __contains__(self, item):
        x, y = item
        return 0 <= x < self.x_max and 0 <= y < self.y_max

    def __getitem__(self, item):
        assert self.__contains__(item)
        x, y = item
        return self.terrain[self._y_max - y - 1, x]

    def __repr__(self):
        return np.array2string(self.terrain)

    @property
    def shape(self):
        return self._x_max, self._y_max

    @property
    def x_max(self):
        return self._x_max
    
    @property
    def y_max(self):
        return self._y_max


class TomJerryGame(dtptb.DTPTBGame):
    def __init__(self, game_config):
        super(TomJerryGame, self).__init__()
        with open(game_config, "r") as file:
            self._game_config = json.load(file)

        self._terrain = Terrain(np.array(self._game_config["terrain"]))
        self._cheese_location = self._game_config["cheese"]["cheese.1"]
        self._walkable_cells = [(x, y) for x in range(self._terrain.x_max) for y in range(self._terrain.y_max) if
                                (self._terrain[x, y] == 1 or self._terrain[x,y] == 2)]
        self._door_locations = [(x, y) for x in range(self._terrain.x_max) for y in range(self._terrain.y_max) if
                                self._terrain[x, y] == 2]
        print(self._walkable_cells)
        self._states = self._construct_states()
        self._final = self._construct_final()
        # self._tom_winning_states = self._construct_tom_winning_states()

    def _construct_states(self):
        """
        State representation: (tom.cell, jerry.cell, door_states, turn)
        :return:
        """
        unique, counts = np.unique(self._terrain.terrain, return_counts=True)
        number_of_doors = counts[2]
        possible_door_states = list(itertools.product([0, 1], repeat=number_of_doors))
        return list(
            filter(self._is_state_valid,
                   itertools.product(self._walkable_cells, self._walkable_cells, possible_door_states,
                                     [CheeseState.TOM_TURN, CheeseState.JERRY_TURN]))
        )

    def _move(self, cell, act, steps=1):
        x, y = cell

        if act == gw_utils.GW_ACT_N:
            return x, y + steps

        elif act == gw_utils.GW_ACT_E:
            return x + steps, y

        elif act == gw_utils.GW_ACT_S:
            return self._move(cell, gw_utils.GW_ACT_N, -steps)

        elif act == gw_utils.GW_ACT_W:
            return self._move(cell, gw_utils.GW_ACT_E, -steps)

        elif act == gw_utils.GW_ACT_STAY:
            return cell

    def states(self):
        return self._states

    def actions(self):
        return [
            "N",
            "S",
            "E",
            "W",
        ]

    def delta(self, state, act):
        # Decouple state
        tom_cell, jerry_cell, door_states, turn = state
        cheese1_pos = tuple(self._game_config["cheese"]["cheese.1"])

        # Base case jerry is at the cheese
        if jerry_cell[0] == cheese1_pos[0] and jerry_cell[1] == cheese1_pos[1]:
            return_state = state

        # Jerry's turn to move
        elif turn == CheeseState.JERRY_TURN:
            new_jerry_cell = self._move(jerry_cell, act)
            new_state = (tom_cell, new_jerry_cell, door_states, CheeseState.TOM_TURN)
            if self._is_state_valid(new_state):
                return_state = new_state
            else:
                # return same state and change turn
                return_state = (tom_cell, jerry_cell, door_states, CheeseState.TOM_TURN)

        # Tom's turn to move
        else:   # if turn == CheeseState.TOM_TURN:
            new_tom_cell = self._move(tom_cell, act)
            door_states_list = list(door_states)
            # if tom moved into a door open it and move him there
            for index, door in enumerate(self._door_locations):
                if new_tom_cell == door:
                    door_states_list[index] = CheeseState.DOOR_OPEN

            new_state = (new_tom_cell, jerry_cell, tuple(door_states_list), CheeseState.JERRY_TURN)
            if self._is_state_valid(new_state):
                return_state = new_state
            else:
                # return same state and change turn
                return_state = (tom_cell, jerry_cell, door_states, CheeseState.JERRY_TURN)

        return return_state

    def _construct_final(self):
        return list(filter(self._is_final_state, self.states()))

    def _is_final_state(self, state):
        tom_cell, jerry_cell, door_states, turn = state
        if jerry_cell[0] == self._cheese_location[0] and jerry_cell[1] == self._cheese_location[1]:
            return True
        else:
            return False

    def final(self, state):
        return state in self._final

    def final_states(self):
        return self._final

    def turn(self, state):
        tom_cell, jerry_cell, door_states, turn = state
        return turn

    def _is_state_valid(self, state):
        tom_cell, jerry_cell, door_states, turn = state

        # if tom or jerry are in a wall
        if self._terrain[tom_cell] == 0 or self._terrain[jerry_cell] == 0:
            return False

        # if jerry or tom is in a CLOSED door
        for index, door in enumerate(self._door_locations):
            if (jerry_cell == door and door_states[index] == CheeseState.DOOR_CLOSED) or \
                    (tom_cell == door and door_states[index] == CheeseState.DOOR_CLOSED):
                return False

        return True

class CheeseState:
    TOM_TURN = 1
    JERRY_TURN = 2
    DOOR_CLOSED = 0
    DOOR_OPEN = 1

    def __init__(self, tom_cell, jerry_cell, door_array, turn):
        self.tom_cell = tom_cell
        self.jerry_cell = jerry_cell
        self.door_array = door_array
        self.turn = turn


if __name__ == '__main__':
    conf = os.path.join(curr_file_path, "saved_games", "game_2023_03_24_18_13.conf")
    print(f"Using configuration file: {conf=}")

    ## Create Base Reachability Game ##
    tom_jerry_game = TomJerryGame(game_config=conf)
    arbitrary_state = random.choice(tom_jerry_game.states())
    print("Executing: game = TomJerryGame(game_config=conf)")
    print(f"Executing: random.choice(game.states())={arbitrary_state}")
    tom_jerry_game.initialize(arbitrary_state)
    base_graph = tom_jerry_game.graphify(pointed=False)
    print("Executing: base_graph = game.graphify(pointed=False)")
    ## Create mapping from arena points to game states ##
    trap_subsets = {}
    for node in base_graph.nodes():
        cell_tom, cell_jerry, states_door, player_turn = base_graph["state"][node]
        if cell_jerry not in trap_subsets:
            trap_subsets[cell_jerry] = []
        trap_subsets[cell_jerry].append(node)
    fake_subsets = trap_subsets
    ## Create a subgraph of the base game hiding P1's winning nodes (win1) and non-rationalizable edges ##
    base_solver = reach.SWinReach(base_graph, tom_jerry_game.final_states(), player=2)
    base_solver.solve()
    # TODO create this subgraph
    tom_jerry_subgraph = gg_graph.SubGraph(base_graph)
    tom_jerry_subgraph.hide_nodes(
        base_solver.winning_nodes(1))
    # TODO ask AK
    #  Should there be any non-rationalizable edges left? Those would just be edges that go from win1 to win2 in the
    #  original game right? Since we removed win1 from this game and are only looking at win2 are all actions rationalizable?

    # Create set of nodes where tom wins (reaches jerry) ##
    tom_win_nodes = list()
    for node in tom_jerry_subgraph.nodes():
        cell_tom, cell_jerry, states_door, player_turn = tom_jerry_subgraph["state"][node]
        if cell_jerry == cell_tom:
            tom_win_nodes.append(node)

    ## Allocate traps and fakes ##
    arena_traps, arena_fakes, covered_states, trap_states, fake_states = solvers.greedy_max(
        tom_jerry_subgraph, trap_subsets=trap_subsets, fake_subsets=fake_subsets, max_traps=1, max_fakes=0)
    ## Create Decoy Allocation Game ##
    decoy_alloc_game = decoy_models.DecoyAllocGame(game=tom_jerry_game, traps=trap_states, fakes=fake_states)
    ## Create P2's Perceptual Game ##
    p2_perceptual_game = decoy_models.PerceptualGameP2(game=tom_jerry_game, traps=trap_states, fakes=fake_states)
    ## Solve p2's perceptual game ##
    solution_p2_perceptual_game = reach.SWinReach(tom_jerry_subgraph, final=tom_jerry_game.final_states())
    solution_p2_perceptual_game.solve()  # P2's strategy
    ## Create Reachability Game of P1 ##
    p1_reachability_game = decoy_models.ReachabilityGameOfP1(
        p2_game=p2_perceptual_game, traps=trap_states, solution_p2_game=solution_p2_perceptual_game)
    ## Create Hypergame ##
    hypergame = decoy_models.Hypergame(  # P1's strategy
        p2_game=p2_perceptual_game, solution_p2_game=solution_p2_perceptual_game, traps=trap_states, fakes=fake_states)

    window = TomJerryWindow(name="Tom and Jerry", size=(660, 480), game_config=conf)
    for trap in arena_traps:
        window.add_trap(trap[0], trap[1])
    for fake_target in arena_fakes:
        window.add_fake(fake_target[0], fake_target[1])
    window.run()