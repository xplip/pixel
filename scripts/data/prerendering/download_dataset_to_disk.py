"""
Basic script that downloads and caches a huggingface dataset to disk
"""

import logging
import sys

from datasets import load_dataset

logger = logging.getLogger(__name__)


def main():
    # Setup logging
    log_level = logging.INFO
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
        level=log_level,
    )
    logger.setLevel(log_level)

    dataset_name = sys.argv[1]
    dataset_split = sys.argv[2]
    cache_dir = sys.argv[3]
    auth_token = sys.argv[4]

    logger.info("Start loading data")
    load_dataset(
        dataset_name,
        split=dataset_split,
        use_auth_token=auth_token,
        cache_dir=cache_dir,
    )
    logger.info("Finished loading data")


if __name__ == "__main__":
    main()
