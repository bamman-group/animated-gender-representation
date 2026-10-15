#!/usr/bin/env bash
# One-shot setup for a fresh checkout on a local GPU server (see the README's
# "Setup" section): download the iCartoonFace data, prepare/verify its layout,
# and fetch a DINOv2 checkout + pretrained weights. Assumes the conda env is
# already created/activated (conda env create -f environment.yml). Steps that
# hit Google Drive can be rate-limited; failures are reported but don't abort
# the rest, so you can re-run after fixing a single step (all steps are
# idempotent).
#
#   ARCH=vitb14 bash scripts/setup.sh     # DINOv2 arch to download (default vitb14)
set -uo pipefail
cd "$(dirname "$0")/.."

echo ">>> [1/3] Downloading iCartoonFace data (Google Drive) ..."
bash scripts/download_data.sh || echo "!! download_data.sh failed - see its manual-download notes."

echo ">>> [2/3] Preparing / verifying data layout ..."
python scripts/prepare_data.py || echo "!! prepare_data.py failed - check data/raw/ contents."

echo ">>> [3/3] Downloading DINOv2 (vitb14, vitl14) checkouts + pretrained weights ..."
bash scripts/download_dino.sh vitb14 || echo "!! download_dino.sh vitb14 failed."
bash scripts/download_dino.sh vitl14 || echo "!! download_dino.sh vitl14 failed."

echo
echo "Setup done. Next:"
echo "  bash scripts/evaluate_all.sh   # out-of-the-box baselines (+ fine-tuned, if present)"
echo "  bash scripts/train_all.sh      # fine-tune both backbones on iCartoonFace identities"
echo "  TEST=1 bash scripts/evaluate_all.sh   # quick end-to-end smoke test first"
