from collections import defaultdict
from pathlib import Path

import numpy as np
import prefltlf2pdfa.viz
import pygame
import scipy.stats

from automatica2025 import *
from ggsolver.simulation.statemachine import *
from utils import load_pickle

logger.remove()
logger.add(sys.stdout, level="ERROR")

OUT_DIR = Path().absolute() / ".tmp"
N_RUNS = 500
CONFIG = {
    "num_columns": 5,
    "num_rows": 4,
    "actions": ["N", "E", "S", "W", "Y", "T"],
    "bee_initial_loc": (1, 0),
    "bird_initial_loc": (3, 1),
    "battery_capacity": 12,
    "bird_bounds": {(2, 0), (2, 1), (3, 0), (3, 1), (4, 0), (4, 1)},
    "tulip_loc": (4, 3),
    "orchid_loc": (1, 1),
    "daisy_loc": (0, 2),
    "bee_dynamic_stochastic": False,
    "bee_dynamic_stochasticity_prob": 0.1,
    "spec_file_path": Path().parent.absolute() / "beerobot.prefltlf"
}


class BeeRobotEnv(Simulator):
    """
    Bee robot environment.
    """

    def __init__(self, game, monitors=None, config=CONFIG, **kwargs):
        super().__init__(game, monitors, **kwargs)
        self.config = config
        self._obstacles = config.get("obstacles", [])
        self._show_automaton = kwargs.get("show_automaton", True)
        self.metadata["render_fps"] = kwargs.get("render_fps", 0.5)
        self.images = {
            "bee": pygame.image.load(Path().absolute() / "assets" / "bee.png"),
            "bird": pygame.image.load(Path().absolute() / "assets" / "bird.png"),
            "tulip": pygame.image.load(Path().absolute() / "assets" / "tulip.png"),
            "orchid": pygame.image.load(Path().absolute() / "assets" / "orchid.png"),
            "daisy": pygame.image.load(Path().absolute() / "assets" / "daisy.png"),
        }

        # Decide size of window and grid cells
        self._max_dim = None
        self._window_size = None
        self._cell_size = None
        self._configure_pygame_window()

    def delta(self, state, action):
        next_states = self.game.delta(state, action)
        states = dict(enumerate(sorted(next_states.keys())))
        probabilities = [next_states[states[st]] for st in states]
        distribution = scipy.stats.rv_discrete(values=(list(states), probabilities))
        return states[distribution.rvs()]

    def _get_reward(self):
        return 0

    def _is_done(self):
        return self.state.game_state.terminated

    def render(self):
        if self.window is None:
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(self._window_size)
            pygame.display.set_caption("Bee Robot Environment")
            self.clock = pygame.time.Clock()

        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        # Create a canvas to render
        canvas = self._render_frame()

        # Update the display
        if self.render_mode == "human":
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
        else:
            return np.transpose(np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2))
        # # Create a canvas to draw on
        # canvas = pygame.Surface((self.grid_size, self.grid_size))
        # canvas.fill((255, 255, 255))  # White background
        #
        # # Draw grid
        # for x in range(self.config["num_columns"] + 1):
        #     pygame.draw.line(canvas, (0, 0, 0), (x * self.cell_size, 0), (x * self.cell_size, self.grid_size))
        # for y in range(self.config["num_rows"] + 1):
        #     pygame.draw.line(canvas, (0, 0, 0), (0, y * self.cell_size), (self.grid_size, y * self.cell_size))
        #
        # # Draw flowers
        # for flower, loc in [("tulip", self.config["tulip_loc"]),
        #                     ("orchid", self.config["orchid_loc"]),
        #                     ("daisy", self.config["daisy_loc"])]:
        #     x, y = loc
        #     canvas.blit(pygame.transform.scale(self.images[flower], (self.cell_size, self.cell_size)),
        #                 (x * self.cell_size, y * self.cell_size))
        #
        # # Draw bee
        # bee_x, bee_y = self.state.game_state.bee_loc
        # canvas.blit(pygame.transform.scale(self.images["bee"], (self.cell_size, self.cell_size)),
        #             (bee_x * self.cell_size, bee_y * self.cell_size))
        #
        # # Draw bird
        # bird_x, bird_y = self.state.game_state.bird_loc
        # canvas.blit(pygame.transform.scale(self.images["bird"], (self.cell_size, self.cell_size)),
        #             (bird_x * self.cell_size, bird_y * self.cell_size))
        #
        # # Update the display
        # self.window.blit(canvas, canvas.get_rect())
        # pygame.display.update()
        # self.clock.tick(30)

    def _render_frame(self):
        # Create a canvas to draw on
        canvas = pygame.Surface(self._window_size)
        canvas.fill((255, 255, 255))  # White background

        # Determine grid origin in window
        grid_width = self.config["num_columns"] * self._cell_size
        grid_height = self.config["num_rows"] * self._cell_size

        # Calculate offsets to center the grid in the leftmost 500x500 region
        grid_origin_x = (500 - grid_width) // 2
        grid_origin_y = (500 - grid_height) // 2

        # Draw grid
        self._draw_bird_bounds(canvas, self._cell_size, grid_origin_in_window=(grid_origin_x, grid_origin_y))
        self._draw_grid(canvas, self._cell_size, grid_origin_in_window=(grid_origin_x, grid_origin_y))
        self._draw_obstacles(canvas, self._cell_size, grid_origin_in_window=(grid_origin_x, grid_origin_y))
        self._draw_flower("tulip", canvas, self._cell_size, grid_origin_in_window=(grid_origin_x, grid_origin_y))
        self._draw_flower("daisy", canvas, self._cell_size, grid_origin_in_window=(grid_origin_x, grid_origin_y))
        self._draw_flower("orchid", canvas, self._cell_size, grid_origin_in_window=(grid_origin_x, grid_origin_y))
        self._draw_bird(canvas, self._cell_size, grid_origin_in_window=(grid_origin_x, grid_origin_y))

        actions = self.game.actions(self.state)
        self._draw_bee(canvas, self._cell_size, actions, grid_origin_in_window=(grid_origin_x, grid_origin_y))

        # Draw automaton
        if self._show_automaton:
            self._draw_automaton(canvas)

        return canvas

    def _configure_pygame_window(self):
        # Assert gridworld dimensions are smaller than (10, 10)
        assert self.config["num_rows"] <= 10 and self.config["num_columns"] <= 10, \
            "Gridworld dimensions must be smaller than or equal to 10x10."

        # Determine the maximum dimension of the gridworld
        max_dim = max(self.config["num_rows"], self.config["num_columns"])

        # Calculate cell size to fit within a 500x500 area
        self._cell_size = 500 // max_dim

        # Set window size based on whether automaton display is enabled
        if self._show_automaton:
            self._window_size = (1500, 500)  # 1500x500 if automaton is shown
        else:
            self._window_size = (500, 500)  # Square window for gridworld only

    def _draw_grid(self, canvas, cell_size, grid_origin_in_window=(0, 0)):
        grid_color = (0, 0, 0)  # Black
        offset_x, offset_y = grid_origin_in_window
        grid_width = self.config["num_columns"] * self._cell_size
        grid_height = self.config["num_rows"] * self._cell_size

        # Draw vertical lines
        for x in range(self.config["num_columns"] + 1):
            pygame.draw.line(
                canvas,
                grid_color,
                (offset_x + x * cell_size, offset_y),
                (offset_x + x * cell_size, offset_y + grid_height),
                1,
            )

        # Draw horizontal lines
        for y in range(self.config["num_rows"] + 1):
            pygame.draw.line(
                canvas,
                grid_color,
                (offset_x, offset_y + y * cell_size),
                (offset_x + grid_width, offset_y + y * cell_size),
                1,
            )

        # for x in range(self._max_dim + 1):
        #     pygame.draw.line(canvas, grid_color, (x * cell_size, 0), (x * cell_size, self._window_size[1]), 1)
        #     pygame.draw.line(canvas, grid_color, (0, x * cell_size), (self._window_size[1], x * cell_size), 1)

    def _draw_obstacles(self, canvas, cell_size, grid_origin_in_window=(0, 0)):
        offset_x, offset_y = grid_origin_in_window
        for obs in self._obstacles:
            x, y = self._transform_coordinates(obs)
            top_left = (x * cell_size + offset_x, y * cell_size + offset_y)
            bottom_right = ((x + 1) * cell_size + offset_x, (y + 1) * cell_size + offset_y)
            top_right = ((x + 1) * cell_size + offset_x, y * cell_size + offset_y)
            bottom_left = (x * cell_size + offset_x, (y + 1) * cell_size + offset_y)
            pygame.draw.line(canvas, (0, 0, 0), top_left, bottom_right, 2)
            pygame.draw.line(canvas, (0, 0, 0), top_right, bottom_left, 2)

    def _draw_bird_bounds(self, canvas, cell_size, grid_origin_in_window=(0, 0)):
        offset_x, offset_y = grid_origin_in_window
        for x, y in self.config["bird_bounds"]:
            x, y = self._transform_coordinates((x, y))
            rect = pygame.Rect(x * cell_size + offset_x, y * cell_size + offset_y, cell_size, cell_size)
            pygame.draw.rect(canvas, (250, 232, 232), rect)

    def _draw_flower(self, param, canvas, cell_size, grid_origin_in_window=(0, 0)):
        offset_x, offset_y = grid_origin_in_window
        x, y = self._transform_coordinates(self.config[f"{param}_loc"])
        flower_image = pygame.transform.scale(self.images[param], (0.75 * cell_size, 0.75 * cell_size))
        offset_img = (cell_size - flower_image.get_width()) // 2
        canvas.blit(flower_image, (x * cell_size + offset_img + offset_x, y * cell_size + offset_img + offset_y))

    def _draw_bird(self, canvas, cell_size, grid_origin_in_window=(0, 0)):
        offset_x, offset_y = grid_origin_in_window
        x, y = self._transform_coordinates((self.state.game_state.bird_x, self.state.game_state.bird_y))
        bird_image = pygame.transform.scale(self.images["bird"], (0.75 * cell_size, 0.75 * cell_size))
        offset_img = (cell_size - bird_image.get_width()) // 2
        canvas.blit(bird_image, (x * cell_size + offset_img + offset_x, y * cell_size + offset_img + offset_y))

    def _draw_bee(self, canvas, cell_size, actions, grid_origin_in_window=(0, 0)):
        offset_x, offset_y = grid_origin_in_window

        # Show bee position
        x, y = self._transform_coordinates((self.state.game_state.bee_x, self.state.game_state.bee_y))
        bee_image = pygame.transform.scale(self.images["bee"], (0.5 * cell_size, 0.5 * cell_size))
        offset_img = (cell_size - bee_image.get_width()) // 2
        canvas.blit(bee_image, (x * cell_size + offset_img + offset_x, y * cell_size + offset_img + offset_y))

        # Render battery counter
        font = pygame.font.Font(None, 24)  # Default font with size 24
        battery_text = font.render(f"{self.state.game_state.battery}", True, (255, 0, 0))  # Red text
        text_x = x * cell_size + offset_img + offset_x + bee_image.get_width() - battery_text.get_width() // 2
        text_y = y * cell_size + offset_img + offset_y - battery_text.get_height() // 2
        canvas.blit(battery_text, (text_x, text_y))

        # Render arrows for each available action
        for action in actions:
            if action == "N":
                start_offset = (cell_size // 2, cell_size // 4)
                end_offset = (cell_size // 2, 0)
                start_pos = (
                    x * cell_size + start_offset[0] + offset_x,
                    y * cell_size + start_offset[1] + offset_y
                )  # Top border
                end_pos = (
                    x * cell_size + end_offset[0] + offset_x,
                    y * cell_size + end_offset[1] + offset_y
                )

                # end_pos = (start_pos[0], start_pos[1] - arrow_length)
                arrow_tip = [
                    (end_pos[0] - 3, end_pos[1] + 6),
                    (end_pos[0] + 3, end_pos[1] + 6),
                    end_pos
                ]
            elif action == "S":
                start_offset = (cell_size // 2, (3 * cell_size) // 4)
                end_offset = (cell_size // 2, cell_size)
                start_pos = (
                    x * cell_size + start_offset[0] + offset_x,
                    y * cell_size + start_offset[1] + offset_y
                )  # Bottom border
                end_pos = (
                    x * cell_size + end_offset[0] + offset_x,
                    y * cell_size + end_offset[1] + offset_y
                )
                arrow_tip = [
                    (end_pos[0] - 3, end_pos[1] - 6),
                    (end_pos[0] + 3, end_pos[1] - 6),
                    end_pos
                ]
            elif action == "E":
                start_offset = ((3 * cell_size) // 4, cell_size // 2)
                end_offset = (cell_size, cell_size // 2)
                start_pos = (
                    x * cell_size + start_offset[0] + offset_x,
                    y * cell_size + start_offset[1] + offset_y
                )  # Bottom border
                end_pos = (
                    x * cell_size + end_offset[0] + offset_x,
                    y * cell_size + end_offset[1] + offset_y
                )
                arrow_tip = [
                    (end_pos[0] - 6, end_pos[1] - 3),
                    (end_pos[0] - 6, end_pos[1] + 3),
                    end_pos
                ]
            elif action == "W":
                start_offset = (cell_size // 4, cell_size // 2)
                end_offset = (0, cell_size // 2)
                start_pos = (
                    x * cell_size + start_offset[0] + offset_x,
                    y * cell_size + start_offset[1] + offset_y
                )  # Bottom border
                end_pos = (
                    x * cell_size + end_offset[0] + offset_x,
                    y * cell_size + end_offset[1] + offset_y
                )

                arrow_tip = [
                    (end_pos[0] + 6, end_pos[1] - 3),
                    (end_pos[0] + 6, end_pos[1] + 3),
                    end_pos
                ]
            else:
                continue

            color = (0, 0, 255)  # Blue color for the arrow
            # Draw the arrow line
            pygame.draw.line(canvas, color, start_pos, end_pos, 2)

            # Draw the arrowhead
            pygame.draw.polygon(canvas, color, arrow_tip)

    def _draw_automaton(self, canvas):
        # Extract current automaton state information
        paut = self.game.aut
        sa_state = self.state.aut_state
        pg_state = None
        for node, data in paut.pref_graph.nodes(data=True):
            if sa_state in data["partition"]:
                pg_state = str(node)
                break

        # Convert automaton to DOT
        sa_dot, pg_dot = prefltlf2pdfa.viz.paut2dot(paut=paut, show_pg_state=True)
        sa_dot.graph_attr.update({"dpi": "300"})
        pg_dot.graph_attr.update({"dpi": "300"})

        # Set all node colors to white
        for node in sa_dot.nodes():
            sa_dot.get_node(node).attr.update({"fillcolor": "white", "style": "filled"})

        # Set the border color of the current state to green
        sa_state = str(sa_state)
        if sa_state in sa_dot.nodes():
            sa_dot.get_node(sa_state).attr.update({"fillcolor": "green", "penwidth": 2})

        # Set all node colors to white
        for node in pg_dot.nodes():
            pg_dot.get_node(node).attr.update({"fillcolor": "white", "style": "filled"})

        # Set the border color of the current state to green
        if sa_state in pg_dot.nodes():
            pg_dot.get_node(pg_state).attr.update({"fillcolor": "green", "penwidth": 2})

        # Generate PNG images
        prefltlf2pdfa.viz.paut2png(sa_dot, pg_dot, fpath=".tmp", fname="automaton.png")

        # Load images
        automaton_sa = pygame.image.load(".tmp/automaton_sa.png")
        automaton_pg = pygame.image.load(".tmp/automaton_pg.png")

        # Show semi-automaton
        max_dim = max(500, automaton_sa.get_width(), automaton_sa.get_height())
        if max_dim > 500:
            aspect_scale = 500 / max_dim
            automaton_sa = pygame.transform.smoothscale_by(
                automaton_sa,
                aspect_scale * 0.9
            )
        x = 502 + (500 - automaton_sa.get_width()) // 2
        y = (500 - automaton_sa.get_height()) // 2
        canvas.blit(automaton_sa, (x, y))  # Centered in the leftmost 500x500 region

        # Show preference graph
        max_dim = max(500, automaton_pg.get_width(), automaton_pg.get_height())
        if max_dim > 500:
            aspect_scale = 500 / max_dim
            automaton_pg = pygame.transform.smoothscale_by(
                automaton_pg,
                aspect_scale * 0.9
            )
        x = 1000 + (500 - automaton_pg.get_width()) // 2
        y = (500 - automaton_pg.get_height()) // 2
        canvas.blit(automaton_pg, (x, y))  # Centered in the leftmost 500x500 region

    def _transform_coordinates(self, cell: List[int]):
        return cell[0], self.config["num_rows"] - cell[1] - 1


def not_terminated(history):
    last_state = history[-1][0].game_state
    if last_state.battery == 0 and not last_state.terminated:
        return False
    return True


def run_simulation(sim, policy, mdp_graph, n_runs=5000, max_steps=15):
    terminal_states = defaultdict(int)

    for _ in tqdm(range(n_runs), "Running simulation..."):
        curr_state, _ = sim.reset()
        n_steps = 0
        terminated = False

        while n_steps < max_steps and not terminated:
            curr_state_id = mdp_graph.get_state_id(curr_state)
            action = policy[curr_state_id]
            curr_state, reward, terminated, truncated, _ = sim.step(action)
            n_steps += 1

        if n_steps >= max_steps:
            logger.error("Not Terminated")
            continue

        terminal_states[curr_state.aut_state] += 1

    return terminal_states


def main():
    # Load game and policy
    mdp_graph = load_pickle(OUT_DIR / "model.pkl")
    solutions = load_pickle(OUT_DIR / "solutions.pkl")
    sol = solutions[4]
    policy = sol._policy

    # Create simulator
    sim = BeeRobotEnv(
        game=mdp_graph,
        monitors={"no-termination": not_terminated},
    )

    # Run simulations
    terminal_states = run_simulation(sim, policy, mdp_graph, n_runs=N_RUNS)

    # Display results
    objective_nodes = [data["partition"] for node, data in mdp_graph.aut.pref_graph.nodes(data=True)]
    print(f"Weights: {sol._weight}")
    print(f"Theoretical Pr(Visiting pref. graph nodes): {sol._sat_probability_of_objectives}")
    print(f"Partition defined by pref. graph nodes: {objective_nodes}")
    print(f"Observed Pr(Visiting pref. graph nodes): { {k: v / N_RUNS for k, v in terminal_states.items()} }")


def main_viz():
    # Load game and policy
    mdp_graph = load_pickle(OUT_DIR / "model.pkl")
    solutions = load_pickle(OUT_DIR / "solutions.pkl")
    sol = solutions[4]
    policy = sol._policy

    # Create simulator
    sim = BeeRobotEnv(
        game=mdp_graph,
        monitors={"no-termination": not_terminated},
        render_mode="human",
    )

    # Step through the simulation and render
    # Reset simulator
    curr_state, _ = sim.reset()
    print(curr_state)
    terminated = False
    max_steps = 15
    n_steps = 0
    while n_steps < max_steps and not terminated:
        sim.render()  # Render the current state
        curr_state_id = mdp_graph.get_state_id(curr_state)
        action = policy[curr_state_id]
        print(action)
        curr_state, reward, terminated, truncated, _ = sim.step(action)
        print(curr_state)
        n_steps += 1

    sim.close()


if __name__ == '__main__':
    # main()
    main_viz()
