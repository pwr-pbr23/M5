#!/bin/bash

if [ ! -d local_mount ]; then
	mkdir local_mount
fi

docker run --platform linux/amd64 --mount type=bind,source=`pwd`/local_mount,destination=/mount -w /home/playground -it api-modelling:latest
