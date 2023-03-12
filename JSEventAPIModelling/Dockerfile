ARG UBUNTU_VER=18.04
ARG CONDA_VER=latest
ARG OS_TYPE=x86_64

FROM --platform=linux/amd64 ubuntu:${UBUNTU_VER}

RUN apt-get update \
	&& apt-get -y install --no-install-recommends python3 python3-pip git unzip ipython3 vim curl wget jq build-essential make
RUN pip3 install --upgrade pip setuptools wheel
# RUN pip3 install numpy scipy pandas matplotlib

# For some reason, matplotlib is crashing when installing with pip3, so I put anaconda to make it work
ARG CONDA_VER
ARG OS_TYPE
# Install miniconda to /miniconda
RUN curl -LO "http://repo.continuum.io/miniconda/Miniconda3-${CONDA_VER}-Linux-${OS_TYPE}.sh"
RUN bash Miniconda3-${CONDA_VER}-Linux-${OS_TYPE}.sh -p /miniconda -b
RUN rm Miniconda3-${CONDA_VER}-Linux-${OS_TYPE}.sh
ENV PATH=/miniconda/bin:${PATH}
RUN conda update -y conda

RUN conda install -c anaconda -y matplotlib
RUN conda install -c anaconda ipython numpy scipy pandas

RUN mkdir -p /home/playground

COPY . /home/playground/

WORKDIR /home/playground

RUN ./build.sh
