import os
import json


def generate_config(nodes: int, topology: str, traps: int, fakes: int, solver_type: str, max_out_degree=None):
    name = f"{solver_type}_exp_n{nodes}_{topology}_t{traps}_f{fakes}"
    path = os.path.join("configurations", f"{name}.json")

    cfg = {
      "name": name,
      "directory": None,
      "overwrite": False,
      "type": solver_type,
      "num_trials": 1,
      "use_multiprocessing": 1,
      "max_traps": traps,
      "max_fakes": fakes,
      "graph": {
        "topology": topology,
        "nodes": nodes,
        "max_out_degree": max_out_degree,
        "hubs": None,
        "save": True,
        "save_png": False
      },
      "log": True,
      "save_intermediate_solutions": False,
      "console_log_level": "info",
      "file_log_level": "debug",
      "report": []
    }

    with open(path, 'w') as file:
        file.write(json.dumps(cfg, indent=4))


def main():
    num_of_nodes = [10, 20, 30, 40, 50]
    num_of_traps = [1, 2, 3]

    for n in num_of_nodes:
        for t in num_of_traps:
            generate_config(nodes=n, topology="hybrid", traps=t, fakes=0, solver_type="enumerative", max_out_degree=10)
            generate_config(nodes=n, topology="hybrid", traps=t, fakes=0, solver_type="greedy", max_out_degree=10)


if __name__ == '__main__':
    main()
