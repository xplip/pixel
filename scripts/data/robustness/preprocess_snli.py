"""
Preprocessing script that corrupts SNLI examples via orthographic attacks (20%, 50%, and 80% severity).
Requires: SNLI dataset files

Example usage:

python preprocess_snli.py \
  --cpu_count=40 \
  --attack="phonetic"

"""

import argparse
import logging
import os
import re
import sys
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path
from timeit import default_timer as timer
from typing import Tuple

import pandas as pd
from adversarial_attacks.attack_api import AdversarialAttacker

logger = logging.getLogger(__name__)

SEVERITY_FLOAT_TO_INT = {
    0.2: 20,
    0.5: 50,
    0.8: 80,
    0: 0,
}


class StoreResults:
    def __init__(self) -> None:
        self.name: str = str()
        self.dd = defaultdict(dict)

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, n: str):
        if n and isinstance(n, str):
            self._name = n

    def update(self, d):
        self.dd[self.name] = {k: v for k, v in d}


class Attacker:
    """
    Based on BERT-Defense (Keller et al., 2021)
    """

    def __init__(self, severity: float, args: argparse.Namespace) -> None:
        self.type_of_attack = args.attack
        self.severity_of_attack = severity
        self.attack = AdversarialAttacker()

    def __call__(self, tup: tuple) -> Tuple:
        idx, sent = tup
        type_and_severity = [(self.type_of_attack, self.severity_of_attack)]
        pert = self.attack.multiattack(sent, type_and_severity)
        return idx, pert


stored_results = StoreResults()


def load_data_pd(split: str, args: argparse.Namespace) -> pd.DataFrame:
    """
    Loads data into pd.DataFrame, discards rows with invalid labels, and renames columns
    """
    df = pd.read_csv(
        os.path.join(args.path_to_data_dir, "snli_1.0", f"snli_1.0_{split}.txt"),
        sep="\t",
        usecols=["gold_label", "sentence1", "sentence2"],
    ).dropna(subset=["gold_label", "sentence1", "sentence2"])
    label_space = ["contradiction", "neutral", "entailment"]
    df = df[df["gold_label"].isin(label_space)]  # discard rows without a valid label
    df = df.rename(
        columns={
            "gold_label": "label",
            "sentence1": "premise_org",  # suffix '_org' to keep track of original text
            "sentence2": "hypothesis_org",
        }
    )
    return df


def mp_attack(idx_sentence: list, severity: float, args: argparse.Namespace) -> dict:
    """
    Perturbs the data using multiprocessing
    """
    attacker = Attacker(severity, args)
    with Pool(processes=args.cpu_count) as pool:
        stored_results.update(pool.map(attacker, idx_sentence))
    return


def perturb_column(df: pd.DataFrame, name: str, severity: float, args: argparse.Namespace) -> pd.DataFrame:
    """
    Attacks the column given by {name}. The results are stored in {stored_results}
    and finally merged onto {df}.
    """
    stored_results.name = name
    idx_sentence = list(zip(df.index, df[name]))
    mp_attack(idx_sentence, severity, args)
    perturbed_df = pd.DataFrame.from_dict(stored_results.dd[name], orient="index", columns=[re.sub("_org", "", name)])
    df = df.join(perturbed_df, how="left")
    return df


def get_path_to_data_dir() -> str:
    """
    Returns path to directory 'data' relative to __file__
    """
    return os.path.join(Path(os.path.dirname(os.path.realpath(__file__))).parents[2], "data")


def main(args: argparse.Namespace):
    if not args.path_to_data_dir:
        args.path_to_data_dir = get_path_to_data_dir()
    logger.info(args)
    for split in ["test", "dev", "train"]:
        for severity in [0.2, 0.5, 0.8]:
            logger.info(f"Preprocess {split} split with severity {severity}")
            # name = f"{args.attack}_{split}_{SEVERITY_FLOAT_TO_INT[severity]}"
            start = timer()
            df = load_data_pd(split, args)
            df = perturb_column(df, "premise_org", severity, args)
            df = perturb_column(df, "hypothesis_org", severity, args)
            logger.info(df.tail())
            out_path = os.path.join(
                args.path_to_data_dir, "robustness", "snli", f"{args.attack}{SEVERITY_FLOAT_TO_INT[severity]}"
            )
            Path(out_path).mkdir(parents=True, exist_ok=True)
            df.to_csv(os.path.join(out_path, f"snli_1.0_{split}.txt"), index=None, sep="\t", encoding="utf-8")
            end = timer()
            logger.info(f"Time: {end - start}")


if __name__ == "__main__":
    log_level = logging.INFO
    logging.basicConfig(level=log_level, handlers=[logging.StreamHandler(sys.stdout)])
    logger.setLevel(log_level)

    available_attacks = [
        "phonetic",
        "full-swap",
        "inner-swap",
        "disemvowel",
        "truncate",
        "keyboard-typo",
        "natural-typo",
        "intrude",
        "segmentation",
        "confusable",
    ]

    parser = argparse.ArgumentParser()
    parser.add_argument("--cpu_count", type=int, help="The number of available cpu nodes", default=1)
    parser.add_argument(
        "--attack",
        type=str,
        help="What type of attack",
        required=True,
        choices=available_attacks,
    )
    parser.add_argument(
        "--path_to_data_dir",
        type=str,
        help="Path to the data directory",
    )
    parsed_args = parser.parse_args()
    main(parsed_args)
