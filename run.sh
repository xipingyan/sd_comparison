#!/bin/bash

# VENV
PYTHEN_ENV=../python-env/bin/activate
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

# numactl -C 0-47 python main.py
# numactl -C 0-383 -m 0,1,2,3,4,5 python main.py
# numactl -N 1 --membind=1 python main.py

numactl -N 0 --membind=0 python3 main.py | tee tmp_0.txt  && 
numactl -N 1 --membind=1 python3 main.py | tee tmp_1.txt  && 
numactl -N 2 --membind=2 python3 main.py | tee tmp_2.txt  &&
numactl -N 3 --membind=3 python3 main.py | tee tmp_3.txt  &&
numactl -N 4 --membind=4 python3 main.py | tee tmp_4.txt  &&
numactl -N 5 --membind=5 python3 main.py | tee tmp_5.txt 