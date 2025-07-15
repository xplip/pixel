# Text Rendering Strategies for Pixel Language Models
This repository contains code for running experiments with the bigrams text rendering strategy. 

All models pretrained with the bigrams rendering strategy are available at [https://huggingface.co/Team-PIXEL](https://huggingface.co/Team-PIXEL?search_models=-bigrams):
- [Team-PIXEL/pixel-tiny-bigrams](https://huggingface.co/Team-PIXEL/pixel-tiny-bigrams)
- [Team-PIXEL/pixel-small-bigrams](https://huggingface.co/Team-PIXEL/pixel-small-bigrams)
- [Team-PIXEL/pixel-base-bigrams](https://huggingface.co/Team-PIXEL/pixel-base-bigrams)

## Setup
We use the same [setup](https://github.com/xplip/pixel?tab=readme-ov-file#setup) as for PIXEL.

The `environment.yaml` file in this repository lists tested library versions for a [Conda](https://docs.conda.io/projects/conda/en/stable/user-guide/install/index.html) setup:
```bash
# Create the environment
conda env create -f environment.yaml

# Activate the environment:
conda activate pixel-bigrams

# Remember to install PIXEL
pip install -e .
```

## Rendering text
The fallback fonts can be downloaded following [these instructions](https://github.com/xplip/pixel/blob/main/.github/FINETUNING.md#downloading-data-and-fallback-fonts) or cloned from [https://github.com/jflotz/fonts](https://github.com/jflotz/fonts).

The code to reproduce Figure 1b can be found under [figures](/figures).

## Pretraining `pixel-base-bigrams`
We use the same pretraining setup as [PIXEL](https://github.com/xplip/pixel/blob/main/.github/PRETRAINING.md).

**Rendered datasets**:
- [Team-PIXEL/rendered-wiki_en-bigrams](https://huggingface.co/datasets/Team-PIXEL/rendered-wiki_en-bigrams/settings)
- [Team-PIXEL/rendered-bookcorpus-bigrams](https://huggingface.co/datasets/Team-PIXEL/rendered-bookcorpus-bigrams/settings)

For more efficient training, we recommend using the optimized pretraining script developed for [PIXEL-M4](https://github.com/ilkerkesen/pixel-m4), which renders the text on-the-fly.

**Text datasets**
- [Team-PIXEL/bigrams_bookcorpus_529](https://huggingface.co/datasets/Team-PIXEL/bigrams_bookcorpus_529)
- [Team-PIXEL/bigrams_wiki-en_529](https://huggingface.co/datasets/Team-PIXEL/bigrams_wiki-en_529)


## Finetuning `pixel-base-bigrams`
This repo provides scripts for running GLUE, NER, QA, and UD with the bigrams rendering strategy. All changes are related to the new renderer. 
See [PIXEL](https://github.com/xplip/pixel/blob/main/.github/FINETUNING.md) for detailed finetuning instructions. 

<details>
  <summary><i>Evaluation to verify installation</i></summary>
Same as for PIXEL, some tasks can get caught in a bad local optimum.
If this happens, try tweaking the learning rate, increasing the batch size, or simply change the random seed. 
We are still refining the training recipe, so feel free to experiment. <br> </br>

First, download the data for NER from Masakhane.
```bash
# Create a folder in which we keep the data
mkdir -p data
  
# MasakhaNER
git clone https://github.com/masakhane-io/masakhane-ner.git data/masakhane-ner

# Download the fallback fonts 
git clone https://github.com/jflotz/fonts.git
```

Then evaluate e.g. on Amharic NER.
This should achieve a test_f1 of 51.2 after 321 epochs. 
```bash
# Settings
export MODEL="Team-PIXEL/pixel-base-bigrams"
export LANG="amh"
export DATA_DIR="data/masakhane-ner/data/${LANG}"
export FALLBACK_FONTS_DIR="fonts/fonts" 
export SEQ_LEN=196
export BSZ=64
export GRAD_ACCUM=1
export LR=3e-5
export NUM_STEPS=15000
export SEED=1

export OUTPUT_DIR="experiments"
export RUN_NAME="$(basename ${MODEL})-${LANG}-${SEQ_LEN}-${BSZ}-${GRAD_ACCUM}-${LR}-${NUM_STEPS}-${SEED}"

mkdir -p ${OUTPUT_DIR}/${RUN_NAME}
 
python scripts/training/run_ner_bigrams.py \
--model_name_or_path=${MODEL} \
--remove_unused_columns=False \
--data_dir=${DATA_DIR} \
--do_train \
--do_eval \
--do_predict \
--dropout_prob=0.1 \
--max_seq_length=${SEQ_LEN} \
--max_steps=${NUM_STEPS} \
--early_stopping \
--early_stopping_patience=5 \
--per_device_train_batch_size=${BSZ} \
--gradient_accumulation_steps=${GRAD_ACCUM} \
--learning_rate=${LR} \
--warmup_steps=100 \
--run_name=${RUN_NAME} \
--output_dir=${OUTPUT_DIR}/${RUN_NAME} \
--overwrite_output_dir \
--overwrite_cache \
--logging_strategy=steps \
--logging_steps=100 \
--evaluation_strategy=steps \
--eval_steps=500 \
--save_strategy=steps \
--save_steps=500 \
--save_total_limit=2 \
--report_to=wandb \
--log_predictions \
--load_best_model_at_end=True \
--metric_for_best_model="eval_f1" \
--fp16 \
--half_precision_backend=auto \
--fallback_fonts_dir=${FALLBACK_FONTS_DIR} \
--seed=${SEED}
```

Or on STS-B.
This should achieve an eval_stsb_spearmanr of 86.6 after 70 epochs. 
```bash
export MODEL="Team-PIXEL/pixel-base-bigrams"
export TASK="stsb"
export FALLBACK_FONTS_DIR="fonts/fonts" 
export POOLING_MODE="mean"
export SEQ_LEN=256
export BSZ=64
export GRAD_ACCUM=1
export LR=1e-5
export NUM_STEPS=15000
export SEED=5

export OUTPUT_DIR="experiments"
export RUN_NAME="$(basename ${MODEL})-${TASK}-${SEQ_LEN}-${BSZ}-${GRAD_ACCUM}-${LR}-${NUM_STEPS}-${SEED}"

mkdir -p ${OUTPUT_DIR}/${RUN_NAME}

python scripts/training/run_glue_bigrams.py \
--task_name=${TASK} \
--model_name_or_path=${MODEL} \
--rendering_backend=${RENDERING_BACKEND} \
--remove_unused_columns=False \
--pooling_mode=${POOLING_MODE} \
--do_train \
--do_eval \
--do_predict \
--dropout_prob=0.1 \
--max_seq_length=${SEQ_LEN} \
--max_steps=${NUM_STEPS} \
--early_stopping \
--early_stopping_patience=5 \
--per_device_train_batch_size=${BSZ} \
--gradient_accumulation_steps=${GRAD_ACCUM} \
--learning_rate=${LR} \
--warmup_steps=100 \
--run_name=${RUN_NAME} \
--overwrite_output_dir \
--overwrite_cache \
--logging_strategy=steps \
--logging_steps=100 \
--evaluation_strategy=steps \
--eval_steps=100 \
--save_strategy=steps \
--save_steps=100 \
--save_total_limit=5 \
--report_to=wandb \
--output_dir=${OUTPUT_DIR}/${RUN_NAME} \
--load_best_model_at_end=True \
--fp16 \
--half_precision_backend=auto \
--metric_for_best_model="eval_spearmanr" \
--fallback_fonts_dir=${FALLBACK_FONTS_DIR} \
--seed=${SEED} 
```

</details>

### Citation & Contact

```bibtex
@inproceedings{lotz-etal-2023-text,
    title = "Text Rendering Strategies for Pixel Language Models",
    author = "Lotz, Jonas  and
      Salesky, Elizabeth  and
      Rust, Phillip  and
      Elliott, Desmond",
    editor = "Bouamor, Houda  and
      Pino, Juan  and
      Bali, Kalika",
    booktitle = "Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing",
    month = "dec",
    year = "2023",
    address = "Singapore",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.emnlp-main.628",
    doi = "10.18653/v1/2023.emnlp-main.628",
    pages = "10155--10172",
}
```

We emphasize that this is experimental research code.

**Contact person:**
Jonas F. Lotz (jonasf.lotz@di.ku.dk)
