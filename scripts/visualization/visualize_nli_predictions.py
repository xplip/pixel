"""
Script to visualize the predictions of a PIXEL model for random NLI examples.
Requirements: A trained PIXELForSequenceClassification with CLS pooling & an NLI dataset.

Other sequence classification tasks like sentiment analysis are currently not supported but this would only require
changing the datasets, labels, and column names.

We use the interpretability method from from https://github.com/hila-chefer/Transformer-MM-Explainability.

Credit:
@InProceedings{Chefer_2021_ICCV,
   author    = {Chefer, Hila and Gur, Shir and Wolf, Lior},
   title     = {Generic Attention-Model Explainability for Interpreting Bi-Modal and Encoder-Decoder Transformers},
   booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
   month     = {October},
   year      = {2021},
   pages     = {397-406}
}

"""

import argparse
import logging
import math
import sys
from typing import Dict, Union

import torch
import wandb
from datasets import load_dataset, ClassLabel
from PIL import Image
from pixel import (
    InterpretablePIXELForSequenceClassification,
    PangoCairoTextRenderer,
    PoolingMode,
    format_img,
    generate_visualization,
    get_attention_mask,
    get_transforms,
    glue_strip_spaces,
    resize_model_embeddings,
)
from torch.utils.data import DataLoader
from transformers import ViTConfig, set_seed

logger = logging.getLogger(__name__)


def main(args: argparse.Namespace):
    # Setup logging
    log_level = logging.INFO
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
        level=log_level,
    )
    logger.setLevel(log_level)

    set_seed(args.seed)

    wandb.init()
    wandb.run.name = args.revision

    if args.dataset_name is not None:
        # Downloading and loading the evaluation nli_dataset from the hub.
        nli_dataset = load_dataset(
            args.dataset_name,
            args.dataset_config_name,
            split="validation",
            use_auth_token=args.auth_token if args.auth_token else None,
        )
    else:
        # Loading from local file
        # CSV/JSON file is needed.
        logger.info(f"Loading local dataset file {args.dataset_file}")

        # Hardcoded based on the format of our robustness data
        if args.dataset_file.endswith(".csv") or args.dataset_file.endswith(".txt"):
            # Loading from local csv file
            nli_dataset = load_dataset("csv", data_files=args.dataset_file, delimiter="\t", split="train")
        else:
            # Loading from local json file
            nli_dataset = load_dataset("json", data_files=args.dataset_file, delimiter="\t", split="train")

    if not isinstance(nli_dataset.features["label"], ClassLabel):
        nli_dataset = nli_dataset.class_encode_column("label")

    # Labels
    label_list = nli_dataset.features["label"].names
    num_labels = len(label_list)
    label_name_to_id = {v: i for i, v in enumerate(label_list)}
    label_id_to_name = {v: k for k, v in label_name_to_id.items()}

    # Set NLI dataset column names
    sentence1_key, sentence2_key = ("premise", "hypothesis")

    config_kwargs = {
        "use_auth_token": args.auth_token if args.auth_token else None,
        "revision": args.revision,
    }

    # Load config and model
    config = ViTConfig.from_pretrained(args.model_name_or_path, num_labels=num_labels, **config_kwargs)
    # Only works with CLS pooling. Models trained with other pooling modes are currently not supported!
    model = InterpretablePIXELForSequenceClassification.from_pretrained(
        args.model_name_or_path, config=config, pooling_mode=PoolingMode.CLS, **config_kwargs
    )

    # Load text renderer
    text_renderer = PangoCairoTextRenderer.from_pretrained(
        args.text_renderer_name_or_path if args.text_renderer_name_or_path else args.model_name_or_path,
        fallback_fonts_dir=args.fallback_fonts_dir,
        **config_kwargs,
    )
    # Optionally resize
    if args.max_seq_length:
        text_renderer.max_seq_length = args.max_seq_length
    resize_model_embeddings(model, text_renderer.max_seq_length)

    # Get image transformations to be applied
    transforms = get_transforms(
        do_resize=True,
        size=(text_renderer.pixels_per_patch, text_renderer.pixels_per_patch * text_renderer.max_seq_length),
    )
    # Formatting function that strips unnecessary whitespace
    formatting_fn = glue_strip_spaces

    def preprocess_function(example: dict):
        result = {}
        if sentence2_key:
            encoding = text_renderer(
                text=(formatting_fn(example[sentence1_key]), formatting_fn(example[sentence2_key]))
            )
        else:
            encoding = text_renderer(text=formatting_fn(example[sentence1_key]))

        result["pixel_values"] = transforms(Image.fromarray(encoding.pixel_values)).unsqueeze(0)
        result["attention_mask"] = get_attention_mask(
            encoding.num_text_patches, seq_length=text_renderer.max_seq_length
        ).unsqueeze(0)

        return result

    def print_top_classes(predictions: torch.Tensor, top_k: int, **kwargs):
        prob = torch.softmax(predictions, dim=1)
        class_indices = predictions.data.topk(top_k, dim=1)[1][0].tolist()
        max_str_len = 0
        class_names = []
        for cls_idx in class_indices:
            class_names.append(label_id_to_name[cls_idx])
            if len(label_id_to_name[cls_idx]) > max_str_len:
                max_str_len = len(label_id_to_name[cls_idx])

        logger.info(f"Top {top_k} classes:")
        for cls_idx in class_indices:
            output_string = f"\t{cls_idx} : {label_id_to_name[cls_idx]}"
            output_string += " " * (max_str_len - len(label_id_to_name[cls_idx])) + "\t\t"
            output_string += "value = {:.3f}\t prob = {:.1f}%".format(predictions[0, cls_idx], 100 * prob[0, cls_idx])
            logger.info(output_string)
        logger.info("\n")

    def process_example(ex: Dict[str, Union[str, int]]):
        if sentence2_key:
            logger.info(f"Sentence 1: {ex[sentence1_key]}")
            logger.info(f"Sentence 2: {ex[sentence2_key]}")
        else:
            logger.info(f"Sentence: {ex[sentence1_key]}")

        inputs = preprocess_function(ex)
        model.eval()
        output = model(**inputs)
        print_top_classes(output["logits"], min(num_labels, 5))
        attn_vis = generate_visualization(
            model,
            inputs,
            image_hw=int(math.sqrt(text_renderer.max_seq_length) * text_renderer.pixels_per_patch),
            patch_size=text_renderer.pixels_per_patch,
        )
        img = wandb.Image(format_img(inputs["pixel_values"]))
        label = label_id_to_name[ex["label"]]
        prediction = label_id_to_name[torch.argmax(output["logits"]).detach().item()]
        vis = wandb.Image(attn_vis)

        return img, label, prediction, vis

    # Randomly select N (max 50) examples from the dataset that will be visualized.
    # Due to a memory leak in the interpretability method, a 24GB GPU cannot encode much more than 50 per run
    nli_dataset = nli_dataset.shuffle().select(range(min(args.num_samples, 50)))
    dataloader = DataLoader(nli_dataset, collate_fn=lambda x: x, shuffle=False, batch_size=1)

    # Process examples one by one
    data = []
    for i, example in enumerate(dataloader):
        data.append(process_example(example[0]))

    # Log results to wandb
    vis_table = wandb.Table(columns=["image", "label", "prediction", "attention_vis"], data=data)
    wandb.log({f"Data": vis_table})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, help="Path to pretrained model")
    parser.add_argument(
        "--text_renderer_name_or_path", type=str, default=None, help="Path to pretrained text renderer"
    )
    parser.add_argument(
        "--dataset_name", type=str, default=None, help="The name of the NLI dataset to use (via the datasets library)."
    )
    parser.add_argument(
        "--dataset_config_name", type=str, default=None, help="Subset of the NLI dataset, e.g language ISO code"
    )
    parser.add_argument("--dataset_file", type=str, default=None, help="Path to a local dataset (.csv or .json).")
    parser.add_argument(
        "--fallback_fonts_dir",
        type=str,
        default=None,
        help="Directory containing fallback fonts for PangoCairoTextRenderer",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=50,
        help="Number of random samples from the dataset to "
        "visualize. Note that the number will be capped "
        "at 50 samples to prevent OOM errors due to "
        "a memory leak in the interpretability method.",
    )
    parser.add_argument("--max_seq_length", type=int, default=None, help="Maximum sequence length")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--auth_token", type=str, default="", help="HuggingFace auth token")
    parser.add_argument("--revision", type=str, default="main", help="HuggingFace branch name / commit ID")
    parsed_args = parser.parse_args()

    if parsed_args.dataset_name is not None:
        pass
    elif parsed_args.dataset_file is None:
        raise ValueError("Need either a dataset file or a dataset name.")

    main(parsed_args)
