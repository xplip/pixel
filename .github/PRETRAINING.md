## Pretraining PIXEL

Here we provide instructions for pretraining PIXEL. You can use these to reproduce our results or train your own PIXEL model on a different dataset.

Our rendered pretraining datasets are available on HuggingFace:
- [Team-PIXEL/rendered-bookcorpus](https://huggingface.co/datasets/Team-PIXEL/rendered-bookcorpus)
- [Team-PIXEL/rendered-wikipedia-english](https://huggingface.co/datasets/Team-PIXEL/rendered-wikipedia-english)

The datasets were created, i.e. prerendered from the respective original text datasets, as described below.

### Prerendering the data

<details>
  <summary><i>Show Instructions</i></summary>
&nbsp;

It's not necessary to prerender the data but it does make things a little faster. We simultaneously rendered and uploaded the data to the huggingface hub. This also works if you have no disk space locally :).

We provide two prerendering scripts, one for the bookcorpus, which we streamed directly from the huggingface hub, and one for Wikipedia, which was available locally as a `txt` file in which one line corresponds to one paragraph and articles are separated by triple newlines. The scripts can also be modified to work with the [Wikipedia dataset](https://huggingface.co/datasets/wikipedia) on the huggingface hub, although the 2018 dump we used will not be accessible that way.
Note: we used the `PyGameTextRenderer` for prerendering and the scripts are currently not compatible with the `PangoCairoTextRenderer`, although this would only require a small change in the code.

Executing these scripts requires installing the modified `datasets` library from our git submodule, in which we added support to push the data to the huggingface hub in chunks.

**Rendering Wikipedia:**
```bash
export DATASET_FILE="en.20180201.txt"
export RENDERER_PATH="configs/renderers/noto_renderer"

python scripts/data/prerendering/prerender_wikipedia.py \
  --renderer_name_or_path=${RENDERER_PATH} \
  --data_path=${DATASET_FILE} \
  --chunk_size=100000 \
  --repo_id="<your_target_huggingface_hub_repo_id>" \
  --split="train" \
  --auth_token="<your_auth_token_with_write_access"
```

**Rendering Bookcorpus:**
```bash
export RENDERER_PATH="configs/renderers/noto_renderer"

python scripts/data/prerendering/prerender_bookcorpus.py \
  --renderer_name_or_path=${RENDERER_PATH} \
  --chunk_size=100000 \
  --repo_id="<your_target_huggingface_hub_repo_id>" \
  --split="train" \
  --auth_token="<your_auth_token_with_write_access"
  ```
    
</details>
    
### Training
We provide a wrapper for pretraining that uses the [submitit](https://github.com/facebookincubator/submitit) to flexibly switch between single and multi-node training setups. You can execute it as follows to use the config `json` files in the [configs folder](https://github.com/xplip/pixel/tree/main/configs) to train PIXEL from scratch with the exact same configuration as described in our paper.
    
```bash
# Training on 2 nodes with 4x 40GB A100 GPUs each took around 8 days

python submitit_pretrain.py \
  --job_dir=../experiments \
  --prototype_config_name=scratch_noto_span0.25-dropout \
  --training_config_name=fp16_apex_bs32 \
  --nodes 2 \
  --ngpus 4 \
  --partition gpu
```
    
You can also directly pass the training arguments to the `submitit` script:
```bash
# Here, we pass the model arguments directly
python submitit_pretrain.py \
  --job_dir=../experiments \
  --text_renderer_name_or_path=configs/renderers/noto_renderer \
  --mask_ratio=0.25 \
  --span_mask \
  --masking_max_span_length=6 \
  --masking_cumulative_span_weights=0.2,0.4,0.6,0.8,0.9,1.0 \
  --dropout_prob=0.1 \
  --training_config_name=fp16_apex_bs32 \
  --nodes 2 \
  --ngpus 4 \
  --partition gpu
    
```
