#!/bin/bash

xhost +local:docker
docker build -t qsvm-container .
