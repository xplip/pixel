python scripts/training/run_pos.py \
  --model_name_or_path="Team-PIXEL/pixel-base-finetuned-pos-ud-vietnamese-vtb" \
  --data_dir="data/ud-treebanks-v2.10/UD_Vietnamese-VTB" \
  --remove_unused_columns=False \
  --output_dir="sanity_check" \
  --do_eval \
  --max_seq_length=256 \
  --overwrite_cache
