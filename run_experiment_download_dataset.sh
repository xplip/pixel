# #!/bin/sh
# # Grid Engine options (lines prefixed with #$)
# $ -N pixel          
# $ -wd /exports/eddie/scratch/s2522559/pixel_project/pixel                 
# # $ -l h_rt=48:00:00 
# $ -l h_vmem=1000G
# #  These options are:
# #  job name: -N
# #  use the current working directory: -cwd
# #  runtime limit of 5 minutes: -l h_rt
# #  memory limit of 1 Gbyte: -l h_vmem
# # Initialise the environment modules
# . /etc/profile.d/modules.sh


# # # Load Python
# # module load python/3.4.3

# # # Run the program
# # python hello.py
# source /exports/eddie/scratch/s2522559/conda/bin/activate pixel

python submitit_pretrain.py \
  --job_dir=../experiments \
  --text_renderer_name_or_path=configs/renderers/noto_renderer \
  --mask_ratio=0.25 \
  --span_mask \
  --masking_max_span_length=6 \
  --masking_cumulative_span_weights=0.2,0.4,0.6,0.8,0.9,1.0 \
  --dropout_prob=0.1 \
  --training_config_name=fp16_apex_bs32 \
  # --nodes 1 \
  # --ngpus 4 \
  # --partition gpu
