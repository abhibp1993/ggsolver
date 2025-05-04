import numpy as np

from beerobot import *
from ggsolver.mdp_prefltlf import *
from utils import save_pickle, load_pickle
from pathlib import Path

OUT_DIR = Path().absolute() / ".tmp"


def solve_for_weights(model, objective, ordering_vector, num_samples=10, corner_cases=True):
    weights = np.random.dirichlet((1,) * len(ordering_vector), num_samples).tolist()
    if corner_cases:
        weights = np.eye(len(ordering_vector)).tolist() + weights

    solutions = []
    for wt in tqdm(weights, desc="Solving for different weights"):
        solver = QuantitativePrefMDPSolver(model, objective, wt, overwrite=True)
        solver.solve()
        solutions.append(solver)

    return solutions


if __name__ == '__main__':
    mdp_graph = load_pickle(OUT_DIR / "model.pkl")
    objective = load_pickle(OUT_DIR / "objective.pkl")
    ordering_vector = load_pickle(OUT_DIR / "ordering_vector.pkl")

    # Solve for multiple weight vectors
    solutions = solve_for_weights(mdp_graph, objective, ordering_vector)
    save_pickle(solutions, OUT_DIR / "solutions.pkl")
