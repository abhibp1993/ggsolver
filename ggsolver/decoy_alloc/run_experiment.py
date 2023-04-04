"""
Runs experiment and generates report based on cfg_dicturation files.
"""

import ggsolver.decoy_alloc.process_config as cfg

import loguru
logger = loguru.logger
logger.remove()


if __name__ == '__main__':
    config = cfg.process_cfg_file("configurations/config1.json")
    logger.success("Configuration loaded successfully.")
