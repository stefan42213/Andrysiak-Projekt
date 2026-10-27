#!/bin/bash
# Pełny pipeline: podział danych -> trening -> ewaluacja -> eksport do Core ML.
# Logi każdego kroku trafiają do outputs/*.log.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p outputs

run() {
    local log="outputs/$1.log"; shift
    echo "==> $* (log: $log)"
    .venv/bin/python -u -W ignore "$@" > "$log" 2>&1
}

run prepare_data prepare_data.py
run train_baseline train.py --model baseline
run train_mobilenet train.py --model mobilenet
run evaluate evaluate.py
run export export_coreml.py
echo "Gotowe. Wyniki: outputs/RAPORT.md, outputs/coreml_parity.json"
