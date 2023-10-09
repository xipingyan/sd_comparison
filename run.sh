#!/bin/bash
PYTHEN_ENV="../../llm_internal_test/python-env/bin/activate"
if [ -f ${PYTHEN_ENV} ]; then
    source ${PYTHEN_ENV}
    numactl -C 144-191 python main.py
else
    PYTHEN_ENV="../llm_internal_test/python-env/bin/activate"
    source ${PYTHEN_ENV}
    numactl -C 48-63 python main.py
fi