FROM ubuntu:18.04

RUN apt-get update \
	&& apt-get -y install --no-install-recommends python3 python3-pip git unzip ipython3 vim
RUN pip3 install numpy scipy pandas matplotlib

RUN mkdir -p /home/playground

COPY . /home/playground/

WORKDIR /home/playground

RUN ./build.sh

