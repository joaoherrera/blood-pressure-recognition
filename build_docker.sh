#!/bin/bash

docker build --build-arg UID=$(id -u) --build-arg GID=$(id -g) -t blood-pressure-recognition .
