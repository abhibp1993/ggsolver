from collections import defaultdict
from pathlib import Path

import scipy.stats

from automatica2025 import *
from ggsolver.simulation.statemachine import *
from utils import load_pickle

logger.remove()
logger.add(sys.stdout, level="ERROR")

OUT_DIR = Path().absolute() / ".tmp"
N_RUNS = 500


class BeeRobotEnv(Simulator):
    """
    Bee robot environment.
    """

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


if __name__ == '__main__':
    main()
