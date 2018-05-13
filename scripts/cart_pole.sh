#!/bin/bash

export PYTHONPATH="$(pwd)"

python src/cart_pole.py \
  --reset_output_dir \
  --output_dir="outputs" \
  --log_every=50 \
  --reset_every=1 \
  --update_every=1 \
  --n_eps=3000 \
  --bl_dec=0.99 \
  "$@"

