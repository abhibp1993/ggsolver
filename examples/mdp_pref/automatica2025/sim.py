from collections import defaultdict
from pathlib import Path

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
        self.size = max(CONFIG["num_rows"], CONFIG["num_columns"])
        self._obstacles = CONFIG.get("obstacles", [])
        self._show_automaton = kwargs.get("show_automaton", True)
        self.metadata["render_fps"] = kwargs.get("render_fps", 0.5)
        self.images = {
            "bee": pygame.image.load(Path().absolute() / "assets" / "bee.png"),
            "bird": pygame.image.load(Path().absolute() / "assets" / "bird.png"),
            "tulip": pygame.image.load(Path().absolute() / "assets" / "tulip.png"),
            "orchid": pygame.image.load(Path().absolute() / "assets" / "orchid.png"),
            "daisy": pygame.image.load(Path().absolute() / "assets" / "daisy.png"),
        }

        if self._show_automaton:
            self.window_size = (1536, 512)
        else:
            self.window_size = (512, 512)
        self.cell_size = self.window_size[1] // self.size

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
            self.window = pygame.display.set_mode(self.window_size)
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
        canvas = pygame.Surface(self.window_size)
        canvas.fill((255, 255, 255))  # White background

        # Draw grid
        self._draw_bird_bounds(canvas, self.cell_size)
        self._draw_grid(canvas, self.cell_size)
        self._draw_obstacles(canvas, self.cell_size)
        self._draw_flower("tulip", canvas, self.cell_size)
        self._draw_flower("daisy", canvas, self.cell_size)
        self._draw_flower("orchid", canvas, self.cell_size)
        self._draw_bird(canvas, self.cell_size)

        actions = self.game.actions(self.state)
        print(actions)
        self._draw_bee(canvas, self.cell_size, actions)

        # Draw automaton
        if self._show_automaton:
            self._draw_automaton(canvas)

        return canvas

    def _draw_grid(self, canvas, cell_size):
        grid_color = (0, 0, 0)  # black
        for x in range(self.size + 1):
            pygame.draw.line(canvas, grid_color, (x * cell_size, 0), (x * cell_size, self.window_size[1]), 1)
            pygame.draw.line(canvas, grid_color, (0, x * cell_size), (self.window_size[1], x * cell_size), 1)

    def _draw_obstacles(self, canvas, cell_size):
        for obs in self._obstacles:
            x, y = obs
            top_left = (x * cell_size, y * cell_size)
            bottom_right = ((x + 1) * cell_size, (y + 1) * cell_size)
            top_right = ((x + 1) * cell_size, y * cell_size)
            bottom_left = (x * cell_size, (y + 1) * cell_size)
            pygame.draw.line(canvas, (0, 0, 0), top_left, bottom_right, 2)
            pygame.draw.line(canvas, (0, 0, 0), top_right, bottom_left, 2)

    def _draw_bird_bounds(self, canvas, cell_size):
        for x, y in self.config["bird_bounds"]:
            rect = pygame.Rect(x * cell_size, y * cell_size, cell_size, cell_size)
            pygame.draw.rect(canvas, (250, 232, 232), rect)

    def _draw_flower(self, param, canvas, cell_size):
        x, y = self.config[f"{param}_loc"]
        flower_image = pygame.transform.scale(self.images[param], (0.75 * cell_size, 0.75 * cell_size))
        offset = (cell_size - flower_image.get_width()) // 2
        canvas.blit(flower_image, (x * cell_size + offset, y * cell_size + offset))

    def _draw_bird(self, canvas, cell_size):
        x, y = self.state.game_state.bird_x, self.state.game_state.bird_y
        bird_image = pygame.transform.scale(self.images["bird"], (0.75 * cell_size, 0.75 * cell_size))
        offset = (cell_size - bird_image.get_width()) // 2
        canvas.blit(bird_image, (x * cell_size + offset, y * cell_size + offset))

    def _draw_bee(self, canvas, cell_size, actions):
        # Show bee position
        x, y = self.state.game_state.bee_x, self.state.game_state.bee_y
        bee_image = pygame.transform.scale(self.images["bee"], (0.75 * cell_size, 0.75 * cell_size))
        offset = (cell_size - bee_image.get_width()) // 2
        canvas.blit(bee_image, (x * cell_size + offset, y * cell_size + offset))

        # Show possible actions
        # Define arrow size
        arrow_size = int(0.25 * cell_size)

        # # Draw arrows for available actions
        # # Render arrows for each available action
        # arrow_length = cell_size // 4  # Adjust arrow length to fit within the circle
        # for action in actions:
        #     start_pos = (x * cell_size + cell_size // 2, y * cell_size + cell_size // 2)
        #     if action == "N":
        #         # end_pos = (start_pos[0], start_pos[1] - arrow_length)
        #         start_pos = (x * cell_size + cell_size // 2, y * cell_size)  # Top border
        #         end_pos = (start_pos[0], start_pos[1] - arrow_length)
        #         arrow_tip = [
        #             (end_pos[0] - 3, end_pos[1] + 6),
        #             (end_pos[0] + 3, end_pos[1] + 6),
        #             end_pos
        #         ]
        #     elif action == "S":
        #         # end_pos = (start_pos[0], start_pos[1] + arrow_length)
        #         start_pos = (x * cell_size + cell_size // 2, y * cell_size)  # Top border
        #         end_pos = (start_pos[0], start_pos[1] - arrow_length)
        #         arrow_tip = [
        #             (end_pos[0] - 3, end_pos[1] - 6),
        #             (end_pos[0] + 3, end_pos[1] - 6),
        #             end_pos
        #         ]
        #     elif action == "E":
        #         # end_pos = (start_pos[0] + arrow_length, start_pos[1])
        #         start_pos = (x * cell_size + cell_size, y * cell_size + cell_size // 2)  # Right border
        #         end_pos = (start_pos[0] + arrow_length, start_pos[1])
        #         arrow_tip = [
        #             (end_pos[0] - 6, end_pos[1] - 3),
        #             (end_pos[0] - 6, end_pos[1] + 3),
        #             end_pos
        #         ]
        #     elif action == "W":
        #         # end_pos = (start_pos[0] - arrow_length, start_pos[1])
        #         start_pos = (x * cell_size, y * cell_size + cell_size // 2)  # Left border
        #         end_pos = (start_pos[0] - arrow_length, start_pos[1])
        #         arrow_tip = [
        #             (end_pos[0] + 6, end_pos[1] - 3),
        #             (end_pos[0] + 6, end_pos[1] + 3),
        #             end_pos
        #         ]
        #     else:
        #         continue
        #
        # color = (0, 0, 255)  # Blue color for the arrow
        # # Draw the arrow line
        # print(start_pos, end_pos)
        # pygame.draw.line(canvas, color, start_pos, end_pos, 2)
        #
        # # Draw the arrowhead
        # pygame.draw.polygon(canvas, color, arrow_tip)

    def _draw_automaton(self, canvas):
        # Extract relevant information
        paut = self.game.aut
        sa_state = self.state.aut_state
        pg_state = None
        for node, data in paut.pref_graph.nodes(data=True):
            if sa_state in data["partition"]:
                pg_state = str(node)
                break

        # Convert automaton to DOT
        sa_dot, pg_dot = prefltlf2pdfa.viz.paut2dot(paut=paut)

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

        # Scale images
        automaton_sa = pygame.transform.scale(
            automaton_sa,
            (min(480, automaton_sa.get_width()), min(480, automaton_sa.get_height()))
        )
        automaton_pg = pygame.transform.scale(
            automaton_pg,
            (min(480, automaton_pg.get_width()), min(480, automaton_pg.get_height()))
        )

        # Update canvas
        # x = 4 * (self.window_size[1]) // 2 - automaton_sa.get_width() // 2
        # y = (self.window_size[1]) // 2 - automaton_sa.get_height() // 2
        canvas.blit(automaton_sa, (self.window_size[1] + 32, 32))  # Top-right corner
        # canvas.blit(automaton_sa, (x, y))  # Top-right corner

        # x = 6 * (self.window_size[1]) // 2 - automaton_sa.get_width() // 2
        # y = (self.window_size[1]) // 2 - automaton_sa.get_height() // 2
        canvas.blit(automaton_pg, (self.window_size[1] * 2 + 32, 32))  # Bottom-right corner
        # canvas.blit(automaton_pg, (x, y))  # Bottom-right corner


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
