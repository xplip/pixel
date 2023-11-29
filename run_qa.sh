# Optional wandb environment vars
module unload cuda
module load cuda/12.1.1

export TRANSFORMERS_OFFLINE=1
export TRANSFORMERS_CACHE=/exports/eddie/scratch/s2522559/cache
export HF_DATASETS_CACHE=/exports/eddie/scratch/s2522559/cache
export WANDB_PROJECT="pixel-experiments"

# Settings
DATASET_NAME="tydiqa"
DATASET_CONFIG_NAME="secondary_task"
TESTSET_NAME="squad"
MODEL="../experiments/nov14-pretrain1/checkpoint-50000" # also works with "bert-base-cased", etc.
FALLBACK_FONTS_DIR="data/fallback_fonts"  # let's say this is where we downloaded the fonts to
SEQ_LEN=400
STRIDE=160
QUESTION_MAX_LEN=128
BSZ=32
GRAD_ACCUM=1
LR=7e-5
SEED=42
NUM_STEPS=20000
  
RUN_NAME="${DATASET_NAME}-pixel-${SEQ_LEN}-${BSZ}-${GRAD_ACCUM}-${LR}-${NUM_STEPS}-${SEED}"
OUTPUT_DIR="../experiments/${RUN_NAME}"

python scripts/training/run_qa.py \
  --model_name_or_path=${MODEL} \
  --dataset_name=${DATASET_NAME} \
  --dataset_config_name=${DATASET_CONFIG_NAME} \
  --testset_name=${TESTSET_NAME} \
  --remove_unused_columns=False \
  --do_train \
  --do_eval \
  --do_predict \
  --dropout_prob=0.1 \
  --max_seq_length=${SEQ_LEN} \
  --question_max_length=${QUESTION_MAX_LEN} \
  --doc_stride=${STRIDE} \
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
  --metric_for_best_model="eval_f1" \
  --fp16 \
  --half_precision_backend=apex \
  --fallback_fonts_dir=${FALLBACK_FONTS_DIR} \
  --seed=${SEED}