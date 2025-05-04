"""
Run experiment
"""
from pathlib import Path
import os
import pandas as pd
from prefltlf2pdfa import PrefLTLf

from navigation import *


def run_exp_navigation():
    # Initialize pandas dataframe to store results
    df = pd.DataFrame(
        columns=[
            "n_rows",
            "n_cols",
            "n_mdp_states",
            "n_mdp_transitions",
            "|semi-aut.nodes|",
            "mdp_build_time",
            "time_solve(strong order)",
            "time_solve(weak order)",
            "time_solve(weak* order)"
        ]
    )

    # Set up the gridworld sweep parameters
    min_n_rows = 3
    min_n_cols = 3
    max_n_rows = 10
    max_n_cols = 10

    for n_rows, n_cols in itertools.product(range(min_n_rows, max_n_rows), range(min_n_cols, max_n_cols)):
        # Instantiate the domain
        domain = NavigationDomain(n_rows, n_cols)

        # Get preference automaton
        paut = get_preference_aut()

        # Define and flatten the product MDP
        prod_mdp = ProductGame(game=domain, aut=paut)
        builder = BuildGameGraph(game_def=prod_mdp, pointed=True, show_report=True, build_labels=False)
        prod_mdp_graph = builder.build()
        prod_mdp_graph = PrefGameGraph(graph=prod_mdp_graph, aut=paut)

        # Construct objective functions for stochastic orders
        wk_objective, _ = build_stochastic_order_objectives(
            prod_mdp_graph,
            ordering_func=stochastic_weak_order
        )
        wk_star_objective, _ = build_stochastic_order_objectives(
            prod_mdp_graph,
            ordering_func=stochastic_weak_star_order
        )
        strong_objective, _ = build_stochastic_order_objectives(
            prod_mdp_graph,
            ordering_func=stochastic_strong_order
        )

        # Solve the product MDP for each type of stochastic order and record time.
        # 1. Select weight vectors (arbitrary)
        wk_weight = np.random.dirichlet((1,) * len(wk_objective), 1).tolist().pop(0)
        wk_star_weight = np.random.dirichlet((1,) * len(wk_star_objective), 1).tolist().pop(0)
        strong_weight = np.random.dirichlet((1,) * len(strong_objective), 1).tolist().pop(0)

        # 2. Construct solver instances
        wk_solver = QuantitativePrefMDPSolver(
            product_mdp=prod_mdp_graph,
            objective=wk_objective,
            weight=wk_weight,
            overwrite=True,
            validate_policy=False
        )
        wk_star_solver = QuantitativePrefMDPSolver(
            product_mdp=prod_mdp_graph,
            objective=wk_star_objective,
            weight=wk_star_weight,
            overwrite=True,
            validate_policy=False
        )
        strong_solver = QuantitativePrefMDPSolver(
            product_mdp=prod_mdp_graph,
            objective=strong_objective,
            weight=strong_weight,
            overwrite=True,
            validate_policy=False
        )

        # 3. Solve the product MDP
        wk_solver.solve()
        wk_star_solver.solve()
        strong_solver.solve()

        # Update dataframe with results
        new_row = pd.DataFrame(
            [{
                "n_rows": n_rows,
                "n_cols": n_cols,
                "n_mdp_states": prod_mdp_graph.number_of_nodes(),
                "n_mdp_transitions": prod_mdp_graph.number_of_edges(),
                "|semi-aut.nodes|": len(prod_mdp_graph.aut.states),
                "mdp_build_time": builder.run_time,
                "time_solve(strong order)": strong_solver.run_time,
                "time_solve(weak order)": wk_solver.run_time,
                "time_solve(weak* order)": wk_star_solver.run_time
            }],
        )
        df = pd.concat([df, new_row], ignore_index=True)

    # Save results
    OUT_DIR = Path().parent / "results"
    if (OUT_DIR / "navigation.csv").exists():
        os.remove(OUT_DIR / "navigation.csv")
    if not OUT_DIR.exists():
        OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_DIR / "navigation.csv", index=False)


def get_preference_aut():
    spec_file_path = Path(__file__).parent / "navigation.prefltlf"
    alphabet = [set(), {"g1"}, {"g2"}, {"g3"}]
    spec = PrefLTLf.from_file(spec_file_path, alphabet=alphabet, auto_complete="minimal")
    aut = spec.translate()
    return aut


if __name__ == "__main__":
    run_exp_navigation()
