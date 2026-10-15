#!/usr/bin/env bash
# Final evaluation protocol:
#   1. for every trained run, select the best checkpoint by AP@0.5 on the
#      icartoon_val split (src/eval/select_best.py; cached in selection.md)
#   2. evaluate that checkpoint - plain and TTA - on the held-out
#      icartoon_test split and on the film set
# Eval-only baselines (released wf checkpoints, YOLOE) skip step 1.
#
# Missing runs are skipped with a warning, so this can be run at any point.
# Results append to results/results.md; detections are saved under results/
# for src/eval/analyze_dets.py. TEST=1 evaluates 1000 images per run and
# selects on 500.
set -uo pipefail
cd "$(dirname "$0")/.."

RESULTS=results/results.md
TEST_FLAG=${TEST:+--test}
SEL_FLAG=${TEST:+--num-images 500}
mkdir -p results
# (the table header is written by src/eval/eval_common.py report() when the
# file is first created)

FAILED=0

run() {  # run <evaluator-module> <name> <model-flag> <model> [extra args...]
    local evaluator=$1 name=$2 model_flag=$3 model=$4; shift 4
    if [ -z "$model" ]; then
        echo "SKIP $name (no run dir / no checkpoint selected)"
        return 0
    fi
    if [ "${model:0:1}" != "@" ] && [ ! -f "$model" ]; then
        echo "SKIP $name (missing: $model)"
        return 0
    fi
    model=${model#@}   # @-prefix = auto-downloading ultralytics weight name
    for eval_set in icartoon_test film; do
        for tta in "" "--tta"; do
            local tag="${name}_${eval_set}${tta:+-tta}"
            echo; echo ">>> $tag"
            python -m "src.eval.$evaluator" "$model_flag" "$model" $tta $TEST_FLAG "$@" \
                --eval-set "$eval_set" --tag "$tag" \
                --results-file "$RESULTS" \
                --save-dets "results/dets_${tag}.json" || FAILED=1
        done
    done
}

best_of() {  # best_of <stack> <run_dir> -> path of icartoon_val-selected checkpoint
    local stack=$1 dir=$2; shift 2
    [ -d "$dir" ] || return 0
    python -m src.eval.select_best --stack "$stack" --run-dir "$dir" \
        $SEL_FLAG "$@" >&2 || return 0
    python -c "import json; print(json.load(open('$dir/selection.json'))['best'])" \
        2>/dev/null || true
}

# ---- trained models: select on icartoon_val, report on icartoon_test/film -------
run evaluate_retinaface retinaface_wf --trained-model weights/Resnet50_Final.pth
run evaluate_retinaface retinaface_icf \
    --trained-model "$(best_of retinaface runs/retinaface/icf)"
run evaluate_retinaface retinaface_wf-icf \
    --trained-model "$(best_of retinaface runs/retinaface/wf_icf)"

for arch in yolo26 rtdetr; do
    for s in wf icf wf_icf; do
        run evaluate_ultralytics "${arch}_${s/_/-}" \
            --weights "$(best_of ultralytics runs/ultralytics/${arch}_${s})"
    done
done

run evaluate_yunet yunet_wf \
    --checkpoint third_party/libfacedetection.train/weights/yunet_n.pth
run evaluate_yunet yunet_icf    --checkpoint "$(best_of yunet runs/yunet/icf)"
run evaluate_yunet yunet_wf-icf --checkpoint "$(best_of yunet runs/yunet/wf_icf)"

# ---- eval-only baselines ---------------------------------------------------------
# YOLOE: the text prompt is a hyperparameter - select it on icartoon_val like
# any checkpoint (candidates in src/eval/select_prompt.py), then report with
# the winner
python -m src.eval.select_prompt $SEL_FLAG >&2 || true
YOLOE_PROMPT=$(python -c "import json; print(json.load(open('runs/yoloe/selection.json'))['best_prompt'])" 2>/dev/null || echo "cartoon face")
echo "YOLOE prompt: $YOLOE_PROMPT"
run evaluate_ultralytics yoloe26l-zeroshot --weights @yoloe-26l-seg.pt \
    --text-prompt "$YOLOE_PROMPT"

echo
echo "==================== RESULTS ===================="
[ -f "$RESULTS" ] && cat "$RESULTS"
exit $FAILED
