#!/usr/bin/env bash
# SiT-SRA unified launcher: train, sample 50K images, build ADM-format NPZ,
# then compute FID/IS/sFID/precision/recall with the bundled SiT evaluator.
#
# Examples:
#   ./launch.sh --skip-eval
#   ./launch.sh --resume-ckpt exps/sit-b2-sra/checkpoints/step-50000.pt --skip-eval
#   ./launch.sh --eval-only --ckpt exps/sit-b2-sra/checkpoints/step-100000.pt

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

GPU="${GPU:-0,1}"
NUM_GPUS="${NUM_GPUS:-2}"
CONFIG="${CONFIG:-configs/sit-b2-sra.yaml}"
EXP_NAME="${EXP_NAME:-sit-b2-sra}"
RESUME_CKPT="${RESUME_CKPT:-}"
CKPT="${CKPT:-}"
EVAL_ONLY=false
SKIP_EVAL=false
SKIP_METRICS=false
METRICS_ONLY=false
SAMPLE_NPZ=""

NUM_FID_SAMPLES="${NUM_FID_SAMPLES:-50000}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-64}"
EVAL_NUM_STEPS="${EVAL_NUM_STEPS:-250}"
# Table 1 SiT-B/2 comparison is explicitly evaluated without CFG.
CFG_SCALE="${CFG_SCALE:-1.0}"
GUIDANCE_HIGH="${GUIDANCE_HIGH:-0.7}"
MODE="${MODE:-sde}"
VAE="${VAE:-ema}"
SEED="${SEED:-0}"
REF_BATCH="${REF_BATCH:-$SCRIPT_DIR/../../SIT/VIRTUAL_imagenet256_labeled.npz}"
EVALUATOR="${EVALUATOR:-$SCRIPT_DIR/../../SIT/evaluations/evaluator.py}"

usage() {
    sed -n '2,9p' "$SCRIPT_DIR/$(basename "${BASH_SOURCE[0]}")"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --config) CONFIG="$2"; shift 2 ;;
        --exp-name) EXP_NAME="$2"; shift 2 ;;
        --gpu) GPU="$2"; shift 2 ;;
        --num-gpus) NUM_GPUS="$2"; shift 2 ;;
        --resume-ckpt) RESUME_CKPT="$2"; shift 2 ;;
        --ckpt) CKPT="$2"; shift 2 ;;
        --eval-only) EVAL_ONLY=true; shift ;;
        --skip-eval) SKIP_EVAL=true; shift ;;
        --skip-metrics) SKIP_METRICS=true; shift ;;
        --metrics-only) METRICS_ONLY=true; shift ;;
        --sample-npz) SAMPLE_NPZ="$2"; shift 2 ;;
        --ref-batch) REF_BATCH="$2"; shift 2 ;;
        --num-fid-samples) NUM_FID_SAMPLES="$2"; shift 2 ;;
        --eval-batch-size) EVAL_BATCH_SIZE="$2"; shift 2 ;;
        --eval-num-steps) EVAL_NUM_STEPS="$2"; shift 2 ;;
        --cfg-scale) CFG_SCALE="$2"; shift 2 ;;
        --guidance-high) GUIDANCE_HIGH="$2"; shift 2 ;;
        --mode) MODE="$2"; shift 2 ;;
        --vae) VAE="$2"; shift 2 ;;
        --seed) SEED="$2"; shift 2 ;;
        --help|-h) usage; exit 0 ;;
        *) echo "未知参数: $1" >&2; usage >&2; exit 2 ;;
    esac
done

if [[ ! -f "$CONFIG" ]]; then
    echo "找不到配置文件: $CONFIG" >&2
    exit 2
fi
if [[ "$EVAL_ONLY" == false && -z "$EXP_NAME" ]]; then
    echo "训练需要 --exp-name <name>" >&2
    exit 2
fi
if [[ "$EVAL_ONLY" == true && -z "$CKPT" ]]; then
    echo "--eval-only 需要 --ckpt <checkpoint.pt>" >&2
    exit 2
fi
if [[ "$METRICS_ONLY" == true && -z "$SAMPLE_NPZ" ]]; then
    echo "--metrics-only 需要 --sample-npz <samples.npz>" >&2
    exit 2
fi

MODEL="$(awk '$1 == "model:" {print $2; exit}' "$CONFIG")"
RESOLUTION="$(awk '$1 == "resolution:" {print $2; exit}' "$CONFIG")"
MODEL="${MODEL:-SiT-B/2}"
RESOLUTION="${RESOLUTION:-256}"
export CUDA_VISIBLE_DEVICES="$GPU"
MASTER_PORT="$((29500 + RANDOM % 1000))"

run_metrics() {
    local sample_npz="$1"
    local sample_stem eval_step eval_dir
    if [[ ! -f "$EVALUATOR" ]]; then
        echo "找不到 evaluator: $EVALUATOR" >&2
        exit 2
    fi
    if [[ ! -f "$REF_BATCH" ]]; then
        echo "找不到 ImageNet-256 reference batch: $REF_BATCH" >&2
        echo "可用 --ref-batch <path> 指定该文件。" >&2
        exit 2
    fi
    if [[ ! -f "$sample_npz" ]]; then
        echo "找不到 sample NPZ: $sample_npz" >&2
        exit 2
    fi
    sample_stem="$(basename "$sample_npz" .npz)"
    if [[ "$sample_stem" =~ step-([0-9]+) ]]; then
        eval_step="${BASH_REMATCH[1]}"
    else
        eval_step="0"
    fi
    eval_dir="$(dirname "$sample_npz")/eval"
    echo "计算 FID / IS / sFID / Precision / Recall..."
    python "$EVALUATOR" \
        --ref_batch "$REF_BATCH" \
        --sample_batch "$sample_npz" \
        --save_path "$eval_dir" \
        --step "$eval_step" \
        --cfg "$CFG_SCALE" \
        --num_steps "$EVAL_NUM_STEPS"
    echo "完成：评估结果在 $eval_dir。"
}

if [[ "$METRICS_ONLY" == true ]]; then
    run_metrics "$SAMPLE_NPZ"
    exit 0
fi

if [[ "$EVAL_ONLY" == false ]]; then
    echo "================================================"
    echo "开始训练: $EXP_NAME"
    echo "GPU: $GPU ($NUM_GPUS GPUs)"
    echo "配置: $CONFIG"
    echo "================================================"

    train_cmd=(accelerate launch --main_process_port "$MASTER_PORT" --num_processes "$NUM_GPUS")
    if [[ "$NUM_GPUS" -gt 1 ]]; then
        train_cmd+=(--multi_gpu)
    fi
    train_cmd+=(train.py --config "$CONFIG" --exp-name "$EXP_NAME" --seed "$SEED")
    if [[ -n "$RESUME_CKPT" ]]; then
        train_cmd+=(--resume-ckpt "$RESUME_CKPT")
    fi
    printf '训练命令:'; printf ' %q' "${train_cmd[@]}"; echo
    "${train_cmd[@]}"

    if [[ "$SKIP_EVAL" == true ]]; then
        echo "训练完成（按 --skip-eval 跳过采样与 NPZ 转换）。"
        exit 0
    fi
    CKPT="$(find "exps/$EXP_NAME/checkpoints" -maxdepth 1 -name '*.pt' -print 2>/dev/null | sort -V | tail -n 1)"
    if [[ -z "$CKPT" ]]; then
        echo "未找到训练 checkpoint: exps/$EXP_NAME/checkpoints" >&2
        exit 1
    fi
fi

sample_dir="$(dirname "$CKPT")"
echo "================================================"
echo "评估 checkpoint: $CKPT"
echo "模型: $MODEL | $NUM_FID_SAMPLES samples | cfg=$CFG_SCALE"
echo "================================================"

torchrun --nproc_per_node="$NUM_GPUS" --master_port="$MASTER_PORT" generate.py \
    --ckpt "$CKPT" \
    --sample-dir "$sample_dir" \
    --model "$MODEL" \
    --resolution "$RESOLUTION" \
    --num-fid-samples "$NUM_FID_SAMPLES" \
    --per-proc-batch-size "$EVAL_BATCH_SIZE" \
    --num-steps "$EVAL_NUM_STEPS" \
    --cfg-scale "$CFG_SCALE" \
    --guidance-high "$GUIDANCE_HIGH" \
    --mode "$MODE" \
    --vae "$VAE" \
    --global-seed "$SEED"

python npz_convert.py \
    --ckpt "$CKPT" \
    --sample-dir "$sample_dir" \
    --model "$MODEL" \
    --resolution "$RESOLUTION" \
    --num-fid-samples "$NUM_FID_SAMPLES" \
    --cfg-scale "$CFG_SCALE" \
    --mode "$MODE" \
    --vae "$VAE" \
    --global-seed "$SEED"

ckpt_stem="$(basename "$CKPT" .pt)"
ckpt_exp_name="$(basename "$(dirname "$(dirname "$CKPT")")")"
sample_npz="$sample_dir/${ckpt_exp_name}-${ckpt_stem}-${RESOLUTION}-vae-${VAE}-cfg-${CFG_SCALE}-seed-${SEED}-${MODE}.npz"

if [[ "$SKIP_METRICS" == true ]]; then
    echo "完成：NPZ 已保存在 $sample_npz（按 --skip-metrics 跳过指标计算）。"
    exit 0
fi
run_metrics "$sample_npz"
