#!/bin/bash
set -e

MODE="${1:-conda}"

DEPS="matplotlib==3.5.0 numpy==1.21.2 scipy==1.7.3 scikit_learn==1.0.2 joblib==1.1.0 tqdm==4.62.0"

if [ "$MODE" = "uv" ]; then
    uv venv .venv --python 3.8
    PIP="uv pip install --python .venv/bin/python"
    $PIP $DEPS
    $PIP .
    echo "Run 'source .venv/bin/activate' to activate the environment, then 'gen2out-demo'."
elif [ "$MODE" = "conda" ]; then
    conda create -n gen2out python=3.8 -y
    PIP="$(conda run -n gen2out which pip)"
    $PIP install $DEPS
    $PIP install .
    echo "Run 'conda activate gen2out' to activate the environment, then 'gen2out-demo'."
else
    echo "Usage: bash run_install.sh [conda|uv]"
    exit 1
fi
