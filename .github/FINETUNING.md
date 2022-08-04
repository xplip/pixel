## Finetuning PIXEL

Here we provide instructions for finetuning PIXEL. You can use these to reproduce our results or train your own PIXEL model on a different dataset. If your dataset is currently not supported and you don't know how to get started (because the renderer is missing a feature, a classification head is missing, etc.) feel free to open an issue about it. 

Requirements:
- An environment set up as described in our main [README.md](https://github.com/xplip/pixel/blob/main/README.md)

### Downloading data and fallback fonts
<details>
  <summary><i>Show Instructions</i></summary>
&nbsp;

#### Fallback fonts
We provide a script to download fallback fonts for the `PangoCairoTextRenderer`. It is not necessary to use fallback fonts because our default `GoNotoCurrent.ttf` font already covers most languages/scripts. The renderer will log warnings if it encounters unknown glyphs. If that happens, you should definitely consider downloading the fallback fonts and passing the folder to the renderer via `--fallback_fonts_dir` so everything is rendered correctly:
  
```bash
python scripts/data/download_fallback_fonts.py <output_dir>
```

#### Data 
Note: For GLUE, QA, and NLI tasks we fully relied on HuggingFace datasets, so downloading them manually is not necessary

```bash
# Create a folder in which we keep the data
mkdir -p data
  
# MasakhaNER
git clone https://github.com/masakhane-io/masakhane-ner.git data/masakhane-ner
  
# UD data for parsing and POS tagging
wget -qO- https://lindat.mff.cuni.cz/repository/xmlui/bitstream/handle/11234/1-4758/ud-treebanks-v2.10.tgz | tar xvz -C data
  
# SNLI for robustness experiments
cd data
wget https://nlp.stanford.edu/projects/snli/snli_1.0.zip
unzip snli_1.0.zip && rm -rf snli_1.0.zip
cd ..
```

</details>

### Sanity Checks
<details>
  <summary><i>Show Instructions</i></summary>
&nbsp;
  
Before training your own models it makes sense to first run an evaluation of our finetuned models to check that everything is set up correctly. Here are some examples for how to do that:

#### POS Tagging
```bash
# Should achieve an eval_accuracy of 86.8 and test_accuracy of 86.0

export DATA_DIR="data/ud-treebanks-v2.10/UD_Vietnamese-VTB"
export MODEL="Team-PIXEL/pixel-base-finetuned-pos-ud-vietnamese-vtb"

python scripts/training/run_pos.py \
  --model_name_or_path=${MODEL} \
  --data_dir=${DATA_DIR} \
  --remove_unused_columns=False \
  --do_eval \
  --do_predict \
  --max_seq_length=256 \
  --output_dir=test-pos \
  --report_to=wandb \
  --log_predictions \
  --overwrite_cache \
  --fallback_fonts_dir=<path_to_your_fallback_fonts_dir  # not necessary here, but good to check that it works
```
  
#### Dependency Parsing
```bash
# Should achieve an eval_las of 82.2 and test_las of 83.9

export DATA_DIR="data/ud-treebanks-v2.10/UD_Coptic-Scriptorium"
export MODEL="Team-PIXEL/pixel-base-finetuned-parsing-ud-coptic-scriptorium"

python scripts/training/run_ud.py \
  --model_name_or_path=${MODEL} \
  --data_dir=${DATA_DIR} \
  --remove_unused_columns=False \
  --do_eval \
  --do_predict \
  --max_seq_length=256 \
  --output_dir=test-ud \
  --report_to=wandb \
  --log_predictions \
  --overwrite_cache \
  --fallback_fonts_dir=<path_to_your_fallback_fonts_dir  # not necessary here, but good to check that it works
```

#### NER
  
```bash
# Should achieve eval_f1 of 55.0 and test_f1 of 49.0

export LANG="amh"
export DATA_DIR="data/masakhane-ner/data/${LANG}"
export MODEL="Team-PIXEL/pixel-base-finetuned-masakhaner-${LANG}"

python scripts/training/run_ner.py \
  --model_name_or_path=${MODEL} \
  --data_dir=${DATA_DIR} \
  --remove_unused_columns=False \
  --do_eval \
  --do_predict \
  --max_seq_length=196 \
  --output_dir=test-ner \
  --report_to=wandb \
  --log_predictions \
  --overwrite_cache \
  --fallback_fonts_dir=<path_to_your_fallback_fonts_dir  # not necessary here, but good to check that it works
```

#### GLUE
  
```bash
# Should achieve eval_sst2_accuracy of 90.3

export TASK_NAME="sst2"
export MODEL="Team-PIXEL/pixel-base-finetuned-sst2"

python scripts/training/run_glue.py \
  --model_name_or_path=${MODEL} \
  --rendering_backend="pygame" \
  --pooling_mode="mean" \
  --task_name=${TASK_NAME} \
  --remove_unused_columns=False \
  --do_eval \
  --do_predict \
  --max_seq_length=256 \
  --output_dir=test-glue \
  --report_to=wandb \
  --log_predictions \
  --overwrite_cache
```
  
#### Question Answering
  
```bash
# Should achieve eval_f1 of 81.6 and eval_exact_match of 71.7

export DATASET_NAME="squad"
export MODEL="Team-PIXEL/pixel-base-finetuned-squadv1"

python scripts/training/run_qa.py \
  --model_name_or_path=${MODEL} \
  --dataset_name=${DATASET_NAME} \
  --remove_unused_columns=False \
  --do_eval \
  --per_device_eval_batch_size=128 \
  --max_seq_length=400 \
  --doc_stride=160 \
  --output_dir=test-qa \
  --report_to=wandb \
  --overwrite_cache \
  --metric_for_best_model=eval_f1 \
  --fallback_fonts_dir=<path_to_your_fallback_fonts_dir  # not necessary here, but good to check that it works
```

</details>


### Training
 
Finetuning PIXEL works almost in the same way as for any other model in the [transformers library](https://github.com/huggingface/transformers). There are, however, a few important differences:
- Instead of a tokenizer, we use a renderer for PIXEL. The two available backends are "pygame" and "pangocairo". The latter has a lot more functionality and is almost always recommended (it is also the default renderer in our finetuning scripts)
- The maximum sequence length in PIXEL needs to have an integer square root, e.g. `256 = 16 * 16` or `400 = 20 * 20`. This is because of how the image is divided into patches

Note: All examples here use grayscale rendering. When using the "pangocairo" backend, you can activate RGB rendering via `--render_rgb`. However, this will make rendering a little slower, so we recommend to only use it when you know you're working with color inputs.

Here are some examples for how to finetune PIXEL:

#### POS Tagging
<details>
  <summary><i>Show Code Snippet</i></summary>
&nbsp;

```bash
# Optional wandb environment vars
export WANDB_PROJECT="pixel-experiments"

# Settings
export TREEBANK="UD_Vietnamese-VTB"
export DATA_DIR="data/ud-treebanks-v2.10/${TREEBANK}"
export FALLBACK_FONTS_DIR="data/fallback_fonts"  # let's say this is where we downloaded the fonts to
export MODEL="Team-PIXEL/pixel-base" # also works with "bert-base-cased", "roberta-base", etc.
export SEQ_LEN=256
export BSZ=64
export GRAD_ACCUM=1
export LR=5e-5
export SEED=42
export NUM_STEPS=15000
  
export RUN_NAME="${TREEBANK}-$(basename ${MODEL})-${SEQ_LEN}-${BSZ}-${GRAD_ACCUM}-${LR}-${NUM_STEPS}-${SEED}"

python scripts/training/run_pos.py \
  --model_name_or_path=${MODEL} \
  --remove_unused_columns=False \
  --data_dir=${DATA_DIR} \
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
  --output_dir=${RUN_NAME} \
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
  --fp16 \
  --half_precision_backend=apex \
  --fallback_fonts_dir=${FALLBACK_FONTS_DIR} \
  --seed=${SEED}
```

</details>

#### Dependency Parsing
<details>
  <summary><i>Show Code Snippet</i></summary>
&nbsp;

```bash
# Optional wandb environment vars
export WANDB_PROJECT="pixel-experiments"

# Settings
export TREEBANK="UD_Coptic-Scriptorium"
export DATA_DIR="data/ud-treebanks-v2.10/${TREEBANK}"
export FALLBACK_FONTS_DIR="data/fallback_fonts"  # let's say this is where we downloaded the fonts to
export MODEL="Team-PIXEL/pixel-base" # also works with "bert-base-cased", "roberta-base", etc.
export SEQ_LEN=256
export BSZ=64
export GRAD_ACCUM=1
export LR=8e-5
export SEED=42
export NUM_STEPS=15000
  
export RUN_NAME="${TREEBANK}-$(basename ${MODEL})-${SEQ_LEN}-${BSZ}-${GRAD_ACCUM}-${LR}-${NUM_STEPS}-${SEED}"

python scripts/training/run_ud.py \
  --model_name_or_path=${MODEL} \
  --remove_unused_columns=False \
  --data_dir=${DATA_DIR} \
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
  --output_dir=${RUN_NAME} \
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
  --metric_for_best_model="eval_las" \
  --fp16 \
  --half_precision_backend=apex \
  --fallback_fonts_dir=${FALLBACK_FONTS_DIR} \
  --seed=${SEED}
```

</details>

#### NER
<details>
  <summary><i>Show Code Snippet</i></summary>
&nbsp;

```bash
# Optional wandb environment vars
export WANDB_PROJECT="pixel-experiments"

# Settings
export LANG="amh"
export DATA_DIR="data/masakhane-ner/data/${LANG}"
export FALLBACK_FONTS_DIR="data/fallback_fonts"  # let's say this is where we downloaded the fonts to
export MODEL="Team-PIXEL/pixel-base" # also works with "bert-base-cased", "roberta-base", etc.
export SEQ_LEN=196
export BSZ=64
export GRAD_ACCUM=1
export LR=5e-5
export SEED=42
export NUM_STEPS=15000
  
export RUN_NAME="${LANG}-$(basename ${MODEL})-${SEQ_LEN}-${BSZ}-${GRAD_ACCUM}-${LR}-${NUM_STEPS}-${SEED}"

python scripts/training/run_ner.py \
  --model_name_or_path=${MODEL} \
  --remove_unused_columns=False \
  --data_dir=${DATA_DIR} \
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
  --output_dir=${RUN_NAME} \
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
```

</details>


#### GLUE
<details>
  <summary><i>Show Code Snippet</i></summary>
&nbsp;

```bash
  
# Note on GLUE: 
# We found that for some of the tasks (e.g. MNLI), PIXEL can get stuck in a bad local optimum
# A clear indicator of this is when the training loss is not decreasing substantially within the first 1k-3k steps
# If this happens, you can tweak the learning rate slightly, increase the batch size,
# change rendering backends, or often even just the random seed
# We are still trying to find the optimal training recipe for PIXEL on these tasks,
# the recipes used in the paper may not be the best ones out there

# Optional wandb environment vars
export WANDB_PROJECT="pixel-experiments"

# Settings
export TASK="sst2"
export MODEL="Team-PIXEL/pixel-base" # also works with "bert-base-cased", "roberta-base", etc.
export RENDERING_BACKEND="pygame"  # Consider trying out both "pygame" and "pangocairo" to see which one works best
export POOLING_MODE="mean" # Can be "mean", "max", "cls", or "pma1" to "pma8"
export SEQ_LEN=256
export BSZ=64
export GRAD_ACCUM=4  # We found that higher batch sizes can sometimes make training more stable
export LR=3e-5
export SEED=42
export NUM_STEPS=15000
  
export RUN_NAME="${TASK}-$(basename ${MODEL})-${POOLING_MODE}-${RENDERING_BACKEND}-${SEQ_LEN}-${BSZ}-${GRAD_ACCUM}-${LR}-${NUM_STEPS}-${SEED}"

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
  --output_dir=${RUN_NAME} \
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
  --fp16 \
  --half_precision_backend=apex \
  --seed=${SEED}
```

</details>

#### Question Answering
<details>
  <summary><i>Show Code Snippet</i></summary>
&nbsp;

Note: If you know you are working with a right-to-left (RTL) language, you can pass `--is_rtl_language` to the QA script to override the automatic base direction check of the renderer

```bash

# Optional wandb environment vars
export WANDB_PROJECT="pixel-experiments"

# Settings
export DATASET_NAME="tydiqa"
export DATASET_CONFIG_NAME="secondary_task"
export MODEL="Team-PIXEL/pixel-base" # also works with "bert-base-cased", etc.
export FALLBACK_FONTS_DIR="data/fallback_fonts"  # let's say this is where we downloaded the fonts to
export SEQ_LEN=400
export STRIDE=160
export QUESTION_MAX_LEN=128
export BSZ=32
export GRAD_ACCUM=1
export LR=7e-5
export SEED=42
export NUM_STEPS=20000
  
export RUN_NAME="${DATASET_NAME}-$(basename ${MODEL})-${SEQ_LEN}-${BSZ}-${GRAD_ACCUM}-${LR}-${NUM_STEPS}-${SEED}"

python scripts/training/run_qa.py \
  --model_name_or_path=${MODEL} \
  --dataset_name=${DATASET_NAME} \
  --dataset_config_name=${DATASET_CONFIG_NAME} \
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
  --output_dir=${RUN_NAME} \
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
```

</details>

#### XNLI Translate-Train-All
<details>
  <summary><i>Show Code Snippet</i></summary>
&nbsp;

```bash

# Optional wandb environment vars
export WANDB_PROJECT="pixel-experiments"

export MODEL="Team-PIXEL/pixel-base" # also works with "bert-base-cased", "bert-base-multilingual-cased", etc.
export POOLING_MODE="cls" # Can be "mean", "max", "cls", or "pma1" to "pma8"
export SEQ_LEN=196 # Must have an integer square root for PIXEL (e.g. 18*18=324)
export FALLBACK_FONTS_DIR="data/fallback_fonts"  # let's say this is where we downloaded the fonts to
export BSZ=128
export GRAD_ACCUM=2
export LR=2e-5
export SEED=42
export NUM_STEPS=50000

export RUN_NAME="xnli-$(basename ${MODEL})-${POOLING_MODE}-${SEQ_LEN}-${BSZ}-${GRAD_ACCUM}-${LR}-${NUM_STEPS}-${SEED}"
python PVE/scripts/training/run_xnli_translate_train_all.py \
  --model_name_or_path=${MODEL} \
  --remove_unused_columns=False \
  --do_train \
  --do_eval \
  --do_predict \
  --pooling_mode=${POOLING_MODE} \
  --pooler_add_layer_norm=True \
  --dropout_prob=0.1 \
  --max_seq_length=${SEQ_LEN} \
  --max_steps=${NUM_STEPS} \
  --num_train_epochs=10 \
  --per_device_train_batch_size=${BSZ} \
  --gradient_accumulation_steps=${GRAD_ACCUM} \
  --learning_rate=${LR} \
  --warmup_steps=1000 \
  --run_name=${RUN_NAME} \
  --output_dir=${RUN_NAME} \
  --overwrite_cache \
  --logging_strategy=steps \
  --logging_steps=100 \
  --evaluation_strategy=steps \
  --eval_steps=1000 \
  --save_strategy=steps \
  --save_steps=1000 \
  --save_total_limit=5 \
  --report_to=wandb \
  --log_predictions \
  --load_best_model_at_end=True \
  --metric_for_best_model="eval_accuracy" \
  --fp16 \
  --half_precision_backend=apex \
  --seed=${SEED} \
  --fallback_fonts_dir=${FALLBACK_FONTS_DIR}
```

</details>

#### Robustness

To replicate the experiments on robustness, first preprocess the data. This makes for an overall shorter training time compared to perturbing the data on the fly.

<details>
  <summary><i>Show Code Snippets</i></summary>
&nbsp;

For POS tagging:
```bash
python scripts/data/robustness/preprocess_pos.py \
  --attack="confusable" \
  --cpu_count=40
```
For SNLI (English):
```bash
python scripts/data/robustness/preprocess_snli.py \
  --attack="confusable" \
  --cpu_count=40
```
</details>

The experiments can then be run by:

<details>
  <summary><i>Show Code Snippet for POS</i></summary>
&nbsp;

```bash

# Optional wandb environment vars
export WANDB_PROJECT="pixel-experiments"

# Settings
export ROBUSTNESS_ATTACK="confusable"
export ROBUSTNESS_SEVERITY=20
export TREEBANK="UD_English-EWT"
export DATA_DIR="data/robustness/pos/${ROBUSTNESS_ATTACK}${ROBUSTNESS_SEVERITY}" 
export FALLBACK_FONTS_DIR="data/fallback_fonts" 
export MODEL="Team-PIXEL/pixel-base"
export SEQ_LEN=256
export BSZ=64
export GRAD_ACCUM=1
export LR=1e-5 
export SEED=42
export NUM_STEPS=15000
  
export RUN_NAME="${ROBUSTNESS_ATTACK}-${ROBUSTNESS_SEVERITY}-$(basename ${MODEL})-${SEQ_LEN}-${BSZ}-${GRAD_ACCUM}-${LR}-${NUM_STEPS}-${SEED}"

python scripts/training/run_pos.py \
  --model_name_or_path=${MODEL} \
  --remove_unused_columns=False \
  --data_dir=${DATA_DIR} \
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
  --output_dir=${RUN_NAME} \
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
  --fallback_fonts_dir=${FALLBACK_FONTS_DIR} \
  --seed=${SEED}
```

</details>

<details>
  <summary><i>Show Code Snippet for SNLI</i></summary>
&nbsp;

```bash

# Optional wandb environment vars
export WANDB_PROJECT="pixel-experiments"

export ROBUSTNESS_ATTACK="confusable"
export ROBUSTNESS_SEVERITY=20
export ROBUSTNESS_DATA="data/robustness/snli"
export MODEL="Team-PIXEL/pixel-base" 
export POOLING_MODE="mean" 
export SEQ_LEN=324 
export FALLBACK_FONTS_DIR="data/fallback_fonts"  
export BSZ=64
export GRAD_ACCUM=1
export LR=1e-5
export SEED=42
export NUM_STEPS=15000

export RUN_NAME="snli-${ROBUSTNESS_ATTACK}-${ROBUSTNESS_SEVERITY}-$(basename ${MODEL})-${POOLING_MODE}-${SEQ_LEN}-${BSZ}-${GRAD_ACCUM}-${LR}-${NUM_STEPS}-${SEED}"
python scripts/training/run_nli.py \
  --model_name_or_path=${MODEL} \
  --remove_unused_columns=False \
  --do_train \
  --do_eval \
  --do_predict \
  --pooling_mode=${POOLING_MODE} \
  --pooler_add_layer_norm=True \
  --dropout_prob=0.1 \
  --max_seq_length=${SEQ_LEN} \
  --max_steps=${NUM_STEPS} \
  --num_train_epochs=10 \
  --per_device_train_batch_size=${BSZ} \
  --gradient_accumulation_steps=${GRAD_ACCUM} \
  --learning_rate=${LR} \
  --warmup_steps=1000 \
  --run_name=${RUN_NAME} \
  --output_dir=${RUN_NAME} \
  --overwrite_cache \
  --logging_strategy=steps \
  --logging_steps=100 \
  --evaluation_strategy=steps \
  --eval_steps=1000 \
  --save_strategy=steps \
  --save_steps=1000 \
  --save_total_limit=5 \
  --report_to=wandb \
  --log_predictions \
  --load_best_model_at_end=True \
  --metric_for_best_model="eval_accuracy" \
  --seed=${SEED} \
  --fallback_fonts_dir=${FALLBACK_FONTS_DIR} \
  --train_file="${ROBUSTNESS_DATA}/${ROBUSTNESS_ATTACK}${ROBUSTNESS_SEVERITY}/snli_1.0_train.txt" \
  --validation_file="${ROBUSTNESS_DATA}/${ROBUSTNESS_ATTACK}${ROBUSTNESS_SEVERITY}/snli_1.0_dev.txt" \
  --test_file="${ROBUSTNESS_DATA}/${ROBUSTNESS_ATTACK}${ROBUSTNESS_SEVERITY}/snli_1.0_test.txt" \
```

</details>