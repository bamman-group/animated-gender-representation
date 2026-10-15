#!/usr/bin/env bash
# Shared defaults for the per-model training scripts. Sourced, never executed.
#
# Every training knob lives here and nowhere else, so running one model's
# script by itself issues exactly the same command train_all.sh would issue
# for it - that equivalence is the point of this file. Override any of them
# from the environment:
#
#   EPOCHS=10 bash scripts/train_yolo26.sh
#   TEST=1 bash scripts/train_all.sh
#
# Each per-model script also takes SETTINGS to restrict which data settings it
# trains (wf = WIDER Face, icf = iCartoonFace, wf_icf = both):
#
#   SETTINGS=icf bash scripts/train_rtdetr.sh
set -euo pipefail

# repo root, relative to the script that sourced this one (all live in scripts/)
cd "$(dirname "${BASH_SOURCE[1]}")/.."

EPOCHS=${EPOCHS:-40}
BATCH=${BATCH:-16}
RTDETR_BATCH=${RTDETR_BATCH:-16}

TEST_FLAG=${TEST:+--test}
