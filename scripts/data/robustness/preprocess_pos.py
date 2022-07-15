"""
Preprocessing script that corrupts POS tagging examples via orthographic attacks (20%, 50%, and 80% severity).
Requires: A dataset file in conllu format, e.g. from Universal Dependencies

Example usage:

python preprocess_pos.py \
  --attack="phonetic" \
  --cpu_count=40

"""

import argparse
import logging
import os
import sys
from dataclasses import dataclass
from multiprocessing import Pool
from pathlib import Path
from timeit import default_timer as timer
from typing import List

import numpy as np
from adversarial_attacks.attack_api import AdversarialAttacker

logger = logging.getLogger(__name__)

SEVERITY_FLOAT_TO_INT = {
    0.2: 20,
    0.5: 50,
    0.8: 80,
    0: 0,
}


@dataclass
class InputExample:
    """
    A single training/test example for POS tagging.
    Args:
        words: list. The words of the sequence.
        labels: (Optional) list. The parts-of-speech labels for each word
    """

    words: List[str]
    labels: List[str]


class Attacker:
    """
    Based on BERT-Defense (Keller et al., 2021)
    """

    def __init__(self, severity: float, args: argparse.Namespace) -> None:
        self.type_of_attack = args.attack
        self.severity_of_attack = severity
        self.attack = AdversarialAttacker()
        self.rng = np.random.default_rng(42)

    def __call__(self, example: InputExample) -> InputExample:
        type_and_severity = [(self.type_of_attack, self.severity_of_attack)]
        if self.type_of_attack in ["visual", "phonetic", "intrude", "confusable"]:
            # attacks at character level
            example.words = [self.attack.multiattack(word, type_and_severity) for word in example.words]
        else:
            # attacks at word level
            example.words = [
                self.attack.multiattack(word, type_and_severity)
                if self.rng.random() < self.severity_of_attack
                else word
                for word in example.words
            ]
        return example


def get_file(data_dir: str, mode: str) -> str:
    """
    Returns path to file
    """
    fp = os.path.join(data_dir, "ud-treebanks-v2.10", "UD_English-EWT", f"en_ewt-ud-{mode}.conllu")
    return fp


def get_path_to_data_dir() -> str:
    """
    Returns path to directory 'data' relative to __file__
    """
    return os.path.join(Path(os.path.dirname(os.path.realpath(__file__))).parents[2], "data")


def mp_attack(data: list, severity: float, args: argparse.Namespace) -> list:
    """
    Perturbs the data using multiprocessing
    """
    attacker = Attacker(severity, args)
    with Pool(processes=args.cpu_count) as pool:
        data = pool.map(attacker, data)
    return data


def read_examples_from_file(args: argparse.Namespace, mode: str) -> List[InputExample]:
    """
    Reads the data from file and returns pairs of 2nd and 4th column
    """
    file_path = get_file(args.path_to_data_dir, mode)
    examples = []

    with open(file_path, "r", encoding="utf-8") as f:
        words: List[str] = []
        labels: List[str] = []
        for line in f.readlines():
            tok = line.strip().split("\t")
            if len(tok) < 2 or line[0] == "#":
                if words:
                    examples.append(InputExample(words=words, labels=labels))
                    words = []
                    labels = []
            if tok[0].isdigit():
                word, label = tok[1], tok[3]
                words.append(word)
                labels.append(label)
        if words:
            examples.append(InputExample(words=words, labels=labels))
    return examples


def write_to_outfile(out_path: str, data: InputExample, mode: str) -> None:
    """
    Saves the perturbed data
    """
    Path(out_path).mkdir(parents=True, exist_ok=True)  # is a directory: dont include the file name
    fp = os.path.join(out_path, f"en_ewt-ud-{mode}.conllu")
    comment = "# Cats and oats"
    col2 = "c2"
    with open(fp, "w", encoding="utf-8") as out:
        for section in data:
            out.write(comment + "\n")
            words, labels = section.words, section.labels
            for e, (word, label) in enumerate(zip(words, labels)):
                out.write(f"{e+1}\t{word}\t{col2}\t{label}\n")
            out.write("\n")


def main(args: argparse.Namespace) -> None:
    if not args.path_to_data_dir:
        args.path_to_data_dir = get_path_to_data_dir()
    logger.info(args)
    for split in ["test", "dev", "train"]:
        for severity in [0.2, 0.5, 0.8]:
            logger.info(f"Preprocess split:{split} with severity {severity}")
            start = timer()
            out_path = os.path.join(
                args.path_to_data_dir, "robustness", "pos", f"{args.attack}{SEVERITY_FLOAT_TO_INT[severity]}"
            )
            data = read_examples_from_file(args, mode=split)
            data = mp_attack(data=data, severity=severity, args=args)
            write_to_outfile(out_path=out_path, data=data, mode=split)
            end = timer()
            logger.info(f"Time: {end - start}")
    return


if __name__ == "__main__":
    log_level = logging.INFO
    logging.basicConfig(level=log_level, handlers=[logging.StreamHandler(sys.stdout)])
    logger.setLevel(log_level)

    available_attacks = [
        "visual",
        "phonetic",
        "full-swap",
        "inner-swap",
        "disemvowel",
        "truncate",
        "keyboard-typo",
        "natural-typo",
        "intrude",
        "confusable",
    ]

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--attack",
        type=str,
        help="What type of attack",
        required=True,
        choices=available_attacks,
        default="inner-swap",
    )
    parser.add_argument("--cpu_count", type=int, help="The number of available cpu nodes", default=1)
    parser.add_argument(
        "--path_to_data_dir",
        type=str,
        help="Path to the data directory",
    )
    parsed_args = parser.parse_args()
    main(parsed_args)
