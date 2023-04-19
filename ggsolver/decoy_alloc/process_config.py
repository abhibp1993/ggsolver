import json
import multiprocessing
import os
import pathlib
import shutil
import sys
import ggsolver.util as util
import loguru
from datetime import datetime
logger = loguru.logger


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
        cfg_dict['directory'] = directory

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
    logger.info(f"The output of experiment will be saved in {directory.resolve()} folder.")

    return cfg_dict


def process_type(cfg_dict):
    """
    Type of solver to use.
    `enumerative` will exhaustively solve all combinations of decoy allocations.
    `greedy` will use greedy approach to iteratively allocate decoys.

    :param cfg_dict:
    :return:
    """
    assert cfg_dict['type'].lower() in ["enumerative", "greedy"], \
        f"Type of network must be in [enumerative, greedy]. Given is {cfg_dict['type'].lower()}."
    logger.info(f"Solver type set to {cfg_dict['type']}.")
    return cfg_dict


def process_multiprocessing(cfg_dict):
    multiprocessing_flag = cfg_dict['use_multiprocessing']

    # If flag is set to `all`, use maximum available CPUs.
    if isinstance(multiprocessing_flag, str) and multiprocessing_flag == "all":
        cfg_dict['use_multiprocessing'] = multiprocessing.cpu_count()
        logger.info(f"Multiprocessing enabled. Using {cfg_dict['use_multiprocessing']} CPUs.")

    # If suggested number of CPUs is larger than available, clamp it.
    elif isinstance(multiprocessing_flag, int):
        if multiprocessing_flag > multiprocessing.cpu_count():
            cfg_dict['use_multiprocessing'] = multiprocessing.cpu_count()
        logger.info(f"Multiprocessing enabled. Using {cfg_dict['use_multiprocessing']} CPUs.")

    else:
        cfg_dict['use_multiprocessing'] = None
        logger.info(f"Multiprocessing disabled.")

    return cfg_dict


def process_graph_params(cfg_dict):
    # Check topology
    topology = cfg_dict['graph']['topology'].lower()
    assert topology in ["mesh", "ring", "star", "tree", "hybrid"], \
        f"Network topology must be in [mesh, ring, star, tree, hybrid]. Given is {topology}."
    logger.info(f"Network topology set to {topology}.")

    # Based on topology, check for other params
    if topology in ["hybrid", "tree", "star"]:
        assert isinstance(cfg_dict['graph']['max_out_degree'], int), \
            f"For hybrid, tree, star topologies, `max_out_degree` must be specified as integer. " \
            f"Received {cfg_dict['graph']['max_out_degree']}."

    if topology == "star":
        assert isinstance(cfg_dict['graph']['hubs'], int), \
            f"For star topology, `hubs` must be specified as integer. " \
            f"Received {cfg_dict['graph']['hubs']}."

    logger.info(f"Graph properties set: "
                f"topology:{topology}, "
                f"max_out_degree:{cfg_dict['graph']['max_out_degree']}, "
                f"hubs:{cfg_dict['graph']['hubs']}")
    return cfg_dict


def process_console_logging(cfg_dict):
    # Console logging output
    # log_format = "{time:HH:mm:ss.SSS} | " \
    #              "<level> {level: <8} </level> | " \
    #              "{name} | " \
    #              "{function}:{line} | " \
    #              "{message}"

    log_format = "<cyan> {time:HH:mm:ss.SSS} </cyan> | " \
                 "<level> {level: <8} </level>| " \
                 "{message}"

    logger.add(sys.stdout,
               level=cfg_dict["console_log_level"].upper(),
               format=log_format,
               colorize=True,
               enqueue=True
               )

    logger.info(f"Console logger initialized to '{cfg_dict['console_log_level'].upper()}'.")
    return cfg_dict


def configure_log_file(cfg_dict):
    if cfg_dict["log"] is True:
        log_format = "{time:HH:mm:ss.SSS} | " \
                     "{level: <8} | " \
                     "{name} | " \
                     "{function}:{line} | " \
                     "{message}"

        logger.add(
            os.path.join(cfg_dict['directory'], f"{cfg_dict['name']}.log"),
            level=cfg_dict['file_log_level'].upper(),
            format=log_format,
            enqueue=True
        )

        logger.info(f"Starting experiment {cfg_dict['name']} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}.")
        logger.info(f"File logger initialized to '{cfg_dict['file_log_level'].upper()}'.")
    else:
        logger.info(f"File logger disabled.")


def process_cfg_file(cfile):
    """
    Loads a configuration from file.

    :param cfile: (str, Path-like) Absolute path of configuration file.
    :return: (dict) Dictionary containing configuration information.
    """
    # Load configuration file
    with open(cfile, 'r') as file:
        # FIXME When the file is not found the program ends with error code 0
        cfg_dict = json.load(file)
        print(util.ColoredMsg.success(f"Configuration file loaded. Running experiment: {cfg_dict['name']}"))

    # Process logging and debug information.
    cfg_dict = process_console_logging(cfg_dict)

    # Process directory
    cfg_dict = process_dir(cfg_dict)

    # Process log output file
    configure_log_file(cfg_dict)

    # Process type information etc.
    cfg_dict = process_type(cfg_dict)

    # Process graph parameters
    cfg_dict = process_graph_params(cfg_dict)

    # Process multiprocessing availability
    cfg_dict = process_multiprocessing(cfg_dict)

    # Save modified configuration
    path = os.path.join(cfg_dict['directory'], f"{cfg_dict['name']}.json")
    with open(path, "w") as file:
        json.dump(cfg_dict, file, indent=2)

    logger.warning("Report generation not processed. Code to be added.")
    return cfg_dict
