"""
Script used to prerender the bookcorpus from https://huggingface.co/datasets/bookcorpusopen
Processes the dataset book-by-book and uploads the rendered examples in chunks to HuggingFace.
Examples are stored and compressed in parquet files.
Relies on a modified version of the datasets library installed through git submodule.
"""

import argparse
import logging
import sys

from datasets import load_dataset
from PIL import Image
from pixel import PyGameTextRenderer, log_example_while_rendering, push_rendered_chunk_to_hub

logger = logging.getLogger(__name__)


def main(args: argparse.Namespace):
    # Load PyGame renderer
    # PangoCairoTextRenderer will be a better choice. We used the PyGame renderer, so we kept this for reproducibility
    text_renderer = PyGameTextRenderer.from_pretrained(args.renderer_name_or_path, use_auth_token=args.auth_token)

    data = {"pixel_values": [], "num_patches": []}
    dataset_stats = {
        "total_uploaded_size": 0,
        "total_dataset_nbytes": 0,
        "total_num_shards": 0,
        "total_num_examples": 0,
        "total_num_words": 0,
    }

    bookcorpus = load_dataset("bookcorpusopen", split="train", streaming=True)

    max_pixels = text_renderer.pixels_per_patch * text_renderer.max_seq_length - 2 * text_renderer.pixels_per_patch
    target_seq_length = max_pixels
    idx = 0
    for book_id, book in enumerate(bookcorpus):

        num_examples = idx
        num_words = dataset_stats["total_num_words"]

        logger.info(f"{book_id}: {book['title']}, {target_seq_length=}px, {num_examples=}, {num_words=}")

        book_text = book["text"].split("\n")

        width = 0
        block = []
        for line in book_text:
            line = line.strip()
            if line:
                dataset_stats["total_num_words"] += len(line.split(" "))

                line_width = text_renderer.font.get_rect(line).width
                if width + line_width >= target_seq_length:
                    idx += 1
                    sequence = " ".join(block)

                    encoding = text_renderer(text=sequence)

                    data["pixel_values"].append(Image.fromarray(encoding.pixel_values))
                    data["num_patches"].append(encoding.num_text_patches)

                    if idx % args.chunk_size == 0:
                        log_example_while_rendering(idx, sequence, encoding.num_text_patches)
                        dataset_stats = push_rendered_chunk_to_hub(args, data, dataset_stats, idx)
                        data = {"pixel_values": [], "num_patches": []}

                    width = line_width
                    block = [line]
                else:
                    block.append(line)
                    width += line_width

        if len(block) > 0:
            idx += 1
            sequence = " ".join(block)
            encoding = text_renderer(text=sequence)

            data["pixel_values"].append(Image.fromarray(encoding.pixel_values))
            data["num_patches"].append(encoding.num_text_patches)

            if idx % args.chunk_size == 0:
                log_example_while_rendering(idx, sequence, encoding.num_text_patches)
                dataset_stats = push_rendered_chunk_to_hub(args, data, dataset_stats, idx)
                data = {"pixel_values": [], "num_patches": []}

    logger.info(f"Total num words in bookcorpus: {dataset_stats['total_num_words']}")


if __name__ == "__main__":
    log_level = logging.INFO
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
        level=log_level,
    )
    logger.setLevel(log_level)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--renderer_name_or_path",
        type=str,
        help="Path or Huggingface identifier of the text renderer",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=100000,
        help="Push data to hub in chunks of N lines",
    )
    parser.add_argument(
        "--max_lines",
        type=int,
        default=-1,
        help="Only look at the first N non-empty lines",
    )
    parser.add_argument("--repo_id", type=str, help="Name of dataset to upload")
    parser.add_argument("--split", type=str, help="Name of dataset split to upload")
    parser.add_argument(
        "--auth_token",
        type=str,
        help="Huggingface auth token with write access to the repo id",
    )

    parsed_args = parser.parse_args()

    main(parsed_args)
