. /etc/profile.d/modules.sh
module unload cuda
module load cuda/12.1.1

python modify_running_script.py \
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
  --training_config_name=fp16_apex_bs32 \
  --nodes 1 \
  --ngpus 1 \
  # --partition gpu
