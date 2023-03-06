FROM abhibp1993/ggsolver:devel

# The version is the same as ggsolver version. It is updated automatically by rebuild.py
LABEL maintainer="Abhishek N. Kulkarni" \
        email="abhi.bp1993@gmail.com" \
		version="0.1.6"


# Install git
RUN apt-get update && \
    RUNLEVEL=1 DEBIAN_FRONTEND=noninteractive \
    apt-get install -y --allow-unauthenticated --no-install-recommends \
    git && \
    apt-get clean


# Download and install latest version of ggsolver. 
RUN rm -rf ggsolver
RUN git clone https://github.com/abhibp1993/ggsolver.git
RUN cd ggsolver && python3 setup.py install --force