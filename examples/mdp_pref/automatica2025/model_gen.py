from prefltlf2pdfa import PrefLTLf

from automatica2025 import *
from ggsolver.game_tsys import *
from utils import save_pickle
from pathlib import Path

# Define a configuration dictionary to simplify input parameters
OUT_DIR = Path().absolute() / ".tmp"
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


def main():
    # Build model and objectives
    model, aut = build_product_mdp(CONFIG)
    objective, ordering_vector = construct_objectives(model, aut)

    # Save model and objectives
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    save_pickle(model, OUT_DIR / "model.pkl")
    save_pickle(objective, OUT_DIR / "objective.pkl")
    save_pickle(ordering_vector, OUT_DIR / "ordering_vector.pkl")

    # Report completion & path of stored files
    print("MDP generation complete. Generated and saved three files:")
    print(f'\t- Product MDP: {OUT_DIR / "model.pkl"}')
    print(f'\t- Stochastic ordering: List of sets of MDP states: {OUT_DIR / "objective.pkl"}')
    print(f'\t- Stochastic ordering: Vector of pref. nodes: {OUT_DIR / "ordering_vector.pkl"}')


def build_product_mdp(config):
    game = BeeRobotGW(config)

    spec_file_path = config["spec_file_path"]
    alphabet = [set(), {"t"}, {"o"}, {"d"}]
    spec = PrefLTLf.from_file(spec_file_path, alphabet=alphabet, auto_complete="minimal")
    aut = spec.translate()

    product_mdp = ProductGame(game=game, aut=aut)
    builder = BuildGameGraph(game_def=product_mdp, pointed=True, show_report=True, build_labels=False)
    prod_graph = builder.build()

    model = PrefGraphGame(graph=prod_graph, aut=aut)

    return model, aut


def construct_objectives(model, aut, ordering_func=stochastic_weak_order):
    ordering = ordering_func(aut.pref_graph)

    ordering_vector = [set(k) for k, v in
                       sorted({tuple(sorted(v)): k for k, v in ordering.items()}.items(), key=lambda x: x[1])]

    objective = [set() for _ in range(len(ordering_vector))]
    for state in model.states():
        for i, aut_obj_i in enumerate(ordering_vector):
            if state.aut_state in aut_obj_i:
                objective[i].add(state)

    return objective, ordering_vector


if __name__ == '__main__':
    main()
