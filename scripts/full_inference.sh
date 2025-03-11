#! /usr/bin/env bash

set -e

mkdir -p logs/success logs/fail

BASENAME="$(basename "$1")"

if [[ -e "logs/success/$BASENAME" ]]; then
    echo "Skipping $BASENAME: already ran inference successfully"
    echo "Delete logs/success/$BASENAME if you want to re-run inference"
    exit 0
fi

ln -sf "$(realpath "$1")" logs/fail/

python3 scripts/estimate_camera.py --video "$1" --static_camera
python3 scripts/estimate_humans.py --video "$1"
python3 scripts/visualize_tram.py --video "$1"

# Move indicator file if everything ran without problems (requires errexit flag)
mv -f logs/fail/"$BASENAME" logs/success/"$BASENAME"
