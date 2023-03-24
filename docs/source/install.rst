Installation Instructions
=========================

`ggsolver` requires Python (>= 3.9) and spot (>=2.8).
The recommended operating system is Debian bullseye or above.

If you want to use a different operating system, docker is the way to go! Please refer to the
docker installation instructions below.


Linux
-----

First, install spot using instructions given at `install spot <https://spot.lrde.epita.fr/install.html>`_.

Then, install `ggsolver`::

    $ git clone https://github.com/abhibp1993/ggsolver.git
    $ cd ggsolver
    $ python3 setup.py install


Docker Image (other OS)
-----------------------

When using some other OS where either spot or ggsolver is not installed,
it is recommended to use `Docker <https://www.docker.com/>`_
images with an Python IDE such as `PyCharm <https://www.jetbrains.com/pycharm/>`_.
Note that using PyCharm is not necessary, but it makes life easy!!


Assuming Docker client is already installed on your OS, `abhibp1993:ggsolver:latest` docker image can be
downloaded by running::

    $ docker pull abhibp1993/ggsolver


The instructions to set up remote interpreter are given at
`Configure a Remote Interpreter using Docker
<https://www.jetbrains.com/help/pycharm/using-docker-as-a-remote-interpreter.html>`_.


.. note::

    The image `abhibp1993/ggsolver:latest` comes preinstalled with the latest version of ggsolver.
    For the development, use `abhibp1993/ggsolver:devel` image.
