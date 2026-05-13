#!/bin/bash

PYTHON_SCRIPT="scripts/sim_c_ldpc_encoder_decoder.py"

nohup .venv/bin/python -u "$PYTHON_SCRIPT" > output.log 2>&1 &
