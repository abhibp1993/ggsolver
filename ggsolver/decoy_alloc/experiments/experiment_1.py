from run_experiment import run_experiment
import ggsolver.decoy_alloc.process_config as cfg

import loguru
logger = loguru.logger
logger.remove()


def main():
    # Load configuration file
    config_file_path = "../configurations/exp_n10_mesh_t1_f0.json"
    config = cfg.process_cfg_file(config_file_path)

    exec_time, ram_used, vod = run_experiment(config)
    logger.success(f"Finished experiment config:{config['name']} with {exec_time=} sec, {ram_used=} bytes, and {vod=}.")



if __name__ == '__main__':
    with logger.catch():
        main()