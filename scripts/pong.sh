#!/bin/bash

export PYTHONPATH="$(pwd)"

python src/pong.py --load_checkpoint=False --render=False

