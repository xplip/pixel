# Optional wandb environment vars
module unload cuda
module load cuda/12.1.1

export TRANSFORMERS_OFFLINE=1
export TRANSFORMERS_CACHE=/exports/eddie/scratch/s2522559/cache
export HF_DATASETS_CACHE=/exports/eddie/scratch/s2522559/cache

# Optional wandb environment vars
export WANDB_PROJECT="pixel-experiments"

# Settings
TASK="qnli"
MODEL="../experiments/nov14-pretrain1/checkpoint-50000" # also works with "bert-base-cased", "roberta-base", etc.
RENDERING_BACKEND="pygame"  # Consider trying out both "pygame" and "pangocairo" to see which one works best
POOLING_MODE="mean" # Can be "mean", "max", "cls", or "pma1" to "pma8"
SEQ_LEN=256
BSZ=64
GRAD_ACCUM=4  # We found that higher batch sizes can sometimes make training more stable
LR=3e-5
SEED=42
NUM_STEPS=15000
  
RUN_NAME="${TASK}-$(basename ${MODEL})-${POOLING_MODE}-${RENDERING_BACKEND}-${SEQ_LEN}-${BSZ}-${GRAD_ACCUM}-${LR}-${NUM_STEPS}-${SEED}"
OUTPUT_DIR="../experiments/${RUN_NAME}"

python scripts/training/run_glue.py \
  --model_name_or_path=${MODEL} \
  --task_name=${TASK} \
  --rendering_backend=${RENDERING_BACKEND} \
  --pooling_mode=${POOLING_MODE} \
  --pooler_add_layer_norm=True \
  --remove_unused_columns=False \
  --do_train \
  --do_eval \
  --do_predict \
  --dropout_prob=0.1 \
  --max_seq_length=${SEQ_LEN} \
  --max_steps=${NUM_STEPS} \
  --num_train_epochs=10 \
  --early_stopping \
  --early_stopping_patience=5 \
  --per_device_train_batch_size=${BSZ} \
  --gradient_accumulation_steps=${GRAD_ACCUM} \
  --learning_rate=${LR} \
  --warmup_steps=100 \
  --run_name=${RUN_NAME} \
  --output_dir=${OUTPUT_DIR} \
  --overwrite_output_dir \
  --overwrite_cache \
  --logging_strategy=steps \
  --logging_steps=100 \
  --evaluation_strategy=steps \
  --eval_steps=500 \
  --save_strategy=steps \
  --save_steps=500 \
  --save_total_limit=5 \
  --report_to=wandb \
  --log_predictions \
  --load_best_model_at_end=True \
  --metric_for_best_model="eval_accuracy" \
  --seed=${SEED}