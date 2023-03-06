from glob import glob
import os
import sys
from setuptools import setup


if sys.version_info[:2] < (3, 9):
    error = (
        f"ggsolver requires Python 3.9 or later ({sys.version_info[0]}.{sys.version_info[1]} detected)."
    )
    sys.stderr.write(error + "\n")
    sys.exit(1)

name = "ggsolver"
description = "Python package containing set-based solvers for synthesis of winning strategies in " \
              "two-player games on graphs"
authors = {
    "abhibp1993": ("Abhishek N. Kulkarni", "abhi.bp1993@gmail.com"),
}
maintainer = "Abhishek N. Kulkarni"
maintainer_email = "abhi.bp1993@gmail.com"
url = "https://akulkarni.me/"
project_urls = {
    "Bug Tracker": "https://github.com/abhibp1993/ggsolver/issues",
    "Documentation": "",
    "Source Code": "https://github.com/abhibp1993/ggsolver",
}
platforms = ["Linux"]
keywords = [
    "ggsolver",
    "Game theory",
    "Reactive synthesis",
]
classifiers = [
    "Development Status :: 2 - Pre-Alpha",
    "Intended Audience :: Education",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
    "Operating System :: POSIX :: Linux",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Programming Language :: Python :: 3 :: Only",
    "Topic :: Software Development :: Libraries :: Python Modules",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
    "Topic :: Scientific/Engineering :: Mathematics",
]

with open("ggsolver/__init__.py") as fid:
    for line in fid:
        if line.startswith("__version__"):
            version = line.strip().split()[-1][1:-1]
            break

packages = [
    "ggsolver",                 # since 0.1.2
    "ggsolver.interfaces",      # since 0.1.2
    "ggsolver.logic",           # since 0.1.5
    "ggsolver.dtptb",           # since 0.1.2
    "ggsolver.mdp",             # since 0.1.5
    "ggsolver.gridworld",       # since 0.1.6
]

install_requires = []

with open("README.md") as fh:
    long_description = fh.read()

if __name__ == "__main__":

    setup(
        name=name,
        version=version,
        maintainer=maintainer,
        maintainer_email=maintainer_email,
        author=authors["abhibp1993"][0],
        author_email=authors["abhibp1993"][1],
        description=description,
        keywords=keywords,
        long_description=long_description,
        platforms=platforms,
        url=url,
        project_urls=project_urls,
        classifiers=classifiers,
        packages=packages,
        # data_files=data,
        # package_data=package_data,
        install_requires=install_requires,
        # extras_require=extras_require,
        python_requires=">=3.9",
        zip_safe=False,
    )
