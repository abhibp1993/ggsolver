"""
Runs experiment and generates report based on cfg_dicturation files.
"""
import json
import os
import pathlib
import shutil
import sys
import datetime
import loguru
import ggsolver.util as util
logger = loguru.logger
logger.remove()


def process_dir(cfg_dict):
    # Get directory to save output
    directory = cfg_dict["directory"]
    if directory is None:
        # Check if default `out` folder exists. If not, create it.
        path = pathlib.Path("out")
        if not path.exists():
            path.mkdir()
            logger.debug(f"{path.resolve()} does not exist. Running mkdir {path.resolve()}. OK")

        # Check how many times given experiment has been run.
        num_exp = sum([1 for sub_dir in path.iterdir() if sub_dir.name.startswith(cfg_dict['name'])])

        # Create new directory for output
        directory = f"out/{cfg_dict['name']}_{num_exp+1}"

        # Update configuration dictionary
        cfg_dict['config'] = cfg_dict

    # Create directory path
    directory = pathlib.Path(directory)

    # If directory exists, check if it should be overwritten
    if cfg_dict["overwrite"] is None:
        cfg_dict["overwrite"] = False
        logger.debug("Overwrite flag set to False.")

    if directory.exists() and cfg_dict["overwrite"]:
        shutil.rmtree(directory)
    elif directory.exists() and not cfg_dict["overwrite"]:
        name = cfg_dict['name']
        raise FileExistsError(f"Given {directory=} to store output of experiment {name=} exists. "
                              f"The overwrite flag is set to {cfg_dict['overwrite']}.")

    # Create output directory
    directory.mkdir()
    logger.success(f"The output of experiment will be saved in {directory.resolve()} folder.")

    return cfg_dict


def process_type(cfg_dict):
    return cfg_dict


def process_multiprocessing(cfg_dict):
    return cfg_dict


def process_graph_params(cfg_dict):
    return cfg_dict


def process_console_logging(cfg_dict):
    # Console logging output
    log_format = "{time:HH:mm:ss.SSS} | " \
                 "<level> {level: <8} </level> | " \
                 "{name} | " \
                 "{function}:{line} |" \
                 "{message}"

    logger.add(sys.stdout,
               level=cfg_dict["console_log_level"].upper(),
               format=log_format,
               colorize=True
               )

    logger.info(f"Console logger initialized to '{cfg_dict['console_log_level'].upper()}'.")
    return cfg_dict


def process_cfg_dict(cfile):
    """
    Loads a configuration from file.

    :param cfile: (str, Path-like) Absolute path of configuration file.
    :return: (dict) Dictionary containing configuration information.
    """
    # Load configuration file
    with open(cfile, 'r') as file:
        cfg_dict = json.load(file)
        print(util.ColoredMsg.success(f"Configuration file loaded. Running experiment: {cfg_dict['name']}"))

    # Process logging and debug information.
    cfg_dict = process_console_logging(cfg_dict)

    # Process directory and filename
    cfg_dict = process_dir(cfg_dict)

    # # Process overwrite flag
    # cfg_dict = process_overwrite(cfg_dict)
    #
    # # Process type information etc.
    # cfg_dict = process_type(cfg_dict)
    #
    # # Process multiprocessing availability
    # cfg_dict = process_multiprocessing(cfg_dict)
    #
    # # Process graph parameters
    # cfg_dict = process_graph_params(cfg_dict)

    return cfg_dict


if __name__ == '__main__':
    config = process_cfg_dict("configurations/config1.json")
