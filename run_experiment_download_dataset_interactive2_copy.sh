. /etc/profile.d/modules.sh
module unload cuda
module load cuda/12.1.1

export TRANSFORMERS_OFFLINE=1
export TRANSFORMERS_CACHE=/exports/eddie/scratch/s2522559/cache
export HF_DATASETS_CACHE=/exports/eddie/scratch/s2522559/cache

python modify_running_script2_copy.py \
  --overwrite_output_dir \
  --job_dir=../experiments \
  --push_to_hub false \
  --streaming true \
  --do_eval false \
  --text_renderer_name_or_path=configs/renderers/noto_renderer \
  --mask_ratio=0.25 \
  --span_mask \
  --masking_max_span_length=6 \
  --masking_cumulative_span_weights=0.2,0.4,0.6,0.8,0.9,1.0 \
  --dropout_prob=0.1 \
  --training_config_name=fp16_apex_bs32_modified \
  --nodes 1 \
  --n_gpu 4 \
  --interactive_job_name nov14-pretrain1 \
  --overwrite_output_dir false
  # --partition gpu
