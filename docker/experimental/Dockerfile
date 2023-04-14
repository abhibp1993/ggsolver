FROM abhibp1993/ggsolver:devel

# The version is the same as ggsolver version. It is updated automatically by rebuild.py
LABEL maintainer="Abhishek N. Kulkarni" \
        email="abhi.bp1993@gmail.com" \
		version="0.1.6"


## Install git
#RUN apt-get update && \
#    RUNLEVEL=1 DEBIAN_FRONTEND=noninteractive \
#    apt-get install -y --allow-unauthenticated --no-install-recommends \
#    git && \
#    apt-get clean
#
## Install OCaml
#RUN apt-get update && \
#    RUNLEVEL=1 DEBIAN_FRONTEND=noninteractive \
#    apt-get install -y --allow-unauthenticated --no-install-recommends \
#    opam && \
#    apt-get clean
#
#RUN opam init -a -y --disable-sandboxing
#RUN opam switch create 4.03.0 \
#    && eval 'opam config env' \
#    && opam install ocamlbuild ocamlfind -y
#RUN opam install TCSLib extlib ocaml-sat-solvers minisat -y
#RUN opam install ounit -y
#
### Install PGSolver
#RUN rm -rf pgsolver
#RUN git clone https://github.com/tcsprojects/pgsolver
#RUN cd pgsolver  \
#    && eval $(opam env) \
#    && ocaml setup.ml -configure \
#    && ocaml setup.ml -build

RUN pip3 install memory-profiler


