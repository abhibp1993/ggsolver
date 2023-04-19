from ggsolver.decoy_alloc.run_experiment import run_experiment
import ggsolver.decoy_alloc.process_config as cfg
import os
import json
import matplotlib.pyplot as plt

import loguru
logger = loguru.logger
logger.remove()


def main():
    # Define parameters
    results = dict()
    num_of_nodes = [10, 20, 30, 40, 50]
    num_of_traps = [1, 2, 3]

    # Run all experiments
    for n in num_of_nodes:
        for t in num_of_traps:
            # Load configuration file
            config_name = f"greedy_exp_n{n}_mesh_t{t}_f{0}"
            config_file_path = os.path.join("configurations", f"{config_name}.json")
            config = cfg.process_cfg_file(config_file_path)
            # Run experiment
            exec_time, ram_used, vod = run_experiment(config)
            results[(n, t)] = {
                "exec_time": exec_time,
                "ram_used": ram_used,
                "vod": vod,
            }
            logger.success(
                f"Finished experiment config:{config['name']} with {exec_time=} sec, {ram_used=} bytes, and {vod=}.")
    # Log results
    path = os.path.join("out", f"greedy_trap_results.json")
    with open(path, 'w') as file:
        file.write(json.dumps(results, indent=4))

    # Calculate average run-times and vod
    average_runtime = dict()
    average_vod = dict()
    for n in num_of_nodes:
        total_runtime = 0
        total_vod = 0
        count = 0
        for t in num_of_traps:
            total_runtime += results[(n, t)]["exec_time"]
            total_vod += results[(n, t)]["vod"]
            count += 1
        average_runtime[n] = total_runtime / count
        average_vod[n] = total_vod / count
    # Create nodes vs time graphs
    x = num_of_nodes
    y = [time for time in average_runtime.values()]

    fig, ax = plt.subplots()
    ax.plot(x, y)
    ax.set_xlabel('Number of Nodes')
    ax.set_ylabel('Runtime (s)')
    ax.set_title('Runtime vs Number of Nodes in Graph')
    ax.set_xticks(num_of_nodes)

    fig.savefig('./out/greedy_trap_runtime_vs_size.png')
    # Create nodes vs memory graphs
    # Create size vs value of deception graph
    x = num_of_nodes
    y = [vod for vod in average_vod.values()]

    fig, ax = plt.subplots()
    ax.plot(x, y)
    ax.set_xlabel('Number of Nodes')
    ax.set_ylabel('Value of Deception')
    ax.set_title('Value of Deception vs Number of Nodes in Graph')
    ax.set_xticks(num_of_nodes)

    fig.savefig('./out/greedy_trap_vod_vs_size.png')

if __name__ == '__main__':
    with logger.catch():
        main()