#!/bin/bash
set -e

cd /workspace/sdpo

# Editable install from mounted source (fast, no-deps since deps are baked in)
pip3 install --no-deps -e . 2>&1 | tail -1

exec "$@"
