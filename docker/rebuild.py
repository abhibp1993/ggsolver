"""
This file should be run outside a docker image. Preferably from within top level `ggsolver` folder.
"""

from datetime import datetime
import logging
import os

dir_ = os.path.dirname(os.path.realpath(__file__))


logging.basicConfig(
    filename=os.path.join(dir_, 'logs', 'rebuild.log'),
    encoding='utf-8',
    level=logging.DEBUG,
    format="[%(levelname)s] %(message)s"
)


class BColors:
    """
    # Reference: https://stackoverflow.com/questions/287871/how-do-i-print-colored-text-to-the-terminal
    """
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


class ColoredMsg:
    @staticmethod
    def ok(msg):
        return f"{BColors.OKCYAN}{msg}{BColors.ENDC}"

    @staticmethod
    def warn(msg):
        return f"{BColors.WARNING}{msg}{BColors.ENDC}"

    @staticmethod
    def success(msg):
        return f"{BColors.OKGREEN}{msg}{BColors.ENDC}"

    @staticmethod
    def error(msg):
        return f"{BColors.FAIL}{msg}{BColors.ENDC}"

    @staticmethod
    def header(msg):
        return f"{BColors.HEADER}{msg}{BColors.ENDC}"


def get_ggsolver_version():
    ggsolver_path = os.path.dirname(dir_)
    with open(os.path.join(ggsolver_path, "ggsolver/__init__.py")) as fid:
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

    logging.info(f"Updated {fpath} with version {version}.")


def build_docker_image(image, fpath, nocache=False):
    if nocache:
        cmd = f"docker build --no-cache -t {image} {fpath}"
    else:
        cmd = f"docker build -t {image} {fpath}"

    print(ColoredMsg.ok(f"Running {cmd}"))
    logging.info(f"Running {cmd}")

    code = os.system(cmd)
    if code == 0:
        print(ColoredMsg.success(f"SUCCESS!! Updated {image} image."))
        logging.info(f"SUCCESS!! Updated {image} image.")
    else:
        print(ColoredMsg.error(f"PROBLEMS!! Did NOT update {image} image."))
        logging.error(f"PROBLEMS!! Did NOT update {image} image.")


def push_docker_image(image):
    cmd = f"docker push {image}"
    print(ColoredMsg.ok(f"Running {cmd}"))
    logging.info(f"Running {cmd}")

    code = os.system(cmd)
    if code == 0:
        print(ColoredMsg.success(f"SUCCESS!! Pushed {image} image."))
        logging.info(f"SUCCESS!! Pushed {image} image.")
    else:
        print(ColoredMsg.error(f"PROBLEMS!! Did NOT push {image} image."))
        logging.info(f"PROBLEMS!! Did NOT push {image} image.")


if __name__ == '__main__':
    version = get_ggsolver_version()
    logging.info(f"\n\n\t\t********** Running dockerfile rebuild script w/ ver. {version} on {datetime.now()} ********** \n\n")

    # Update dockerfiles
    update_dockerfile(fpath=os.path.join(dir_, "devel/Dockerfile"), version=version)
    update_dockerfile(fpath=os.path.join(dir_, "latest/Dockerfile"), version=version)

    # It's important to build `devel` first since `latest` is based on it.
    build_docker_image(image="abhibp1993/ggsolver:devel", fpath=os.path.join(dir_, "devel"))
    push_docker_image(image="abhibp1993/ggsolver:devel")

    build_docker_image(image="abhibp1993/ggsolver:latest", fpath=os.path.join(dir_, "latest"), nocache=True)
    push_docker_image(image="abhibp1993/ggsolver:latest")

