#!/bin/bash

# VENV
PYTHEN_ENV=./python-env/bin/activate
if [ ! -f ${PYTHEN_ENV} ]; then
    echo "Error: $PYTHEN_ENV not exist."
    exit
fi
echo "PYTHEN_ENV=$PYTHEN_ENV"
source ${PYTHEN_ENV}

# OpenVINO path:
OV_ENV=../openvino/build/install/setupvars.sh
if [ ! -f ${OV_ENV} ]; then
    echo "Error: $OV_ENV not exist."
    exit
fi
echo "OV_ENV=$OV_ENV"
source ${OV_ENV}

numactl -N 1 --membind=1 python main.py