#!/bin/bash
PYTHEN_ENV="../python-env/bin/activate"
if [ ! -f ${PYTHEN_ENV} ]; then
    PYTHEN_ENV="../llm_internal_test/python-env/bin/activate"
fi

echo "PYTHEN_ENV=$PYTHEN_ENV"
source ${PYTHEN_ENV}
source /home/xiping/openvino/build/install/setupvars.sh

# numactl -C 0-47 python main.py
# numactl -C 0-383 -m 0,1,2,3,4,5 python main.py
numactl -N 1 --membind=1 python main.py