"""
This file should be run outside a docker image. Preferably from within top level `ggsolver` folder.
"""

from datetime import datetime
from loguru import logger
import sys
import os
import pathlib

dir_ = pathlib.Path(__file__).parent.absolute()


logger.remove()
logger.add(sys.stdout, level="INFO")
logger.add(os.path.join(dir_, 'build_logs', 'rebuild.log'), level="INFO", rotation="500 MB")


def get_ggsolver_version():
    ggsolver_path = dir_.parent.absolute()
    with open(os.path.join(ggsolver_path, "ggsolver/version.py")) as fid:
        for line in fid:
            if line.startswith("__version__"):
                version_ = line.strip().split()[-1][1:-1]
                break
    return version_


def update_dockerfile(fpath, version):
    with open(fpath, "r") as file:
        lines = file.readlines()
        lines[5] = f'\t\tversion="{version}"\n'

    with open(fpath, "w") as file:
        file.writelines(lines)

    logger.info(f"Updated {fpath} with version {version}.")


def build_docker_image(image, fpath, nocache=False):
    if nocache:
        cmd = f"docker build --no-cache -t {image} {fpath}"
    else:
        cmd = f"docker build -t {image} {fpath}"

    logger.info(f"Running {cmd}")

    code = os.system(cmd)
    if code == 0:
        logger.info(f"SUCCESS!! Updated {image} image.")
    else:
        logger.error(f"PROBLEMS!! Did NOT update {image} image. "
                     f"Did you, by mistake, try to build the image within docker?")

    return code == 0


def push_docker_image(image):
    cmd = f"docker push {image}"
    logger.info(f"Running {cmd}")

    code = os.system(cmd)
    if code == 0:
        logger.info(f"SUCCESS!! Pushed {image} image.")
    else:
        logger.info(f"PROBLEMS!! Did NOT push {image} image. "
                    f"Did you, by mistake, try to build the image within docker?")

    return code == 0


if __name__ == '__main__':
    version = get_ggsolver_version()
    logger.info(f"********** Running dockerfile rebuild script w/ ver. {version} on {datetime.now()} "
                f"using {sys.executable}**********")

    # Update dockerfiles
    update_dockerfile(fpath=os.path.join(dir_, "devel/Dockerfile"), version=version)
    update_dockerfile(fpath=os.path.join(dir_, "latest/Dockerfile"), version=version)

    # It's important to build_logs `devel` first since `latest` is based on it.
    success = build_docker_image(image="devel", fpath=os.path.join(dir_, "devel"))
    if not success:
        logger.error("Code terminated because 'devel' could not be built.")
        sys.exit(1)

    success = build_docker_image(image="abhibp1993/ggsolver:devel", fpath=os.path.join(dir_, "devel"))
    if not success:
        logger.error("Code terminated because 'abhibp1993/ggsolver:devel' could not be built.")
        sys.exit(1)

    success = push_docker_image(image="abhibp1993/ggsolver:devel")
    if not success:
        logger.error("Code terminated because 'abhibp1993/ggsolver:devel' could not be pushed.")
        sys.exit(1)

    success = build_docker_image(image="latest", fpath=os.path.join(dir_, "latest"), nocache=True)
    if not success:
        logger.error("Code terminated because 'latest' could not be built.")
        sys.exit(1)

    success = build_docker_image(image="abhibp1993/ggsolver:latest", fpath=os.path.join(dir_, "latest"), nocache=True)
    if not success:
        logger.error("Code terminated because 'abhibp1993/ggsolver:latest' could not be built.")
        sys.exit(1)

    success = push_docker_image(image="abhibp1993/ggsolver:latest")
    if not success:
        logger.error("Code terminated because 'abhibp1993/ggsolver:latest' could not be pushed.")
        sys.exit(1)

    logger.success("abhibp1993/ggsolver:devel and abhibp1993/ggsolver:latest updated and pushed successfully...")
