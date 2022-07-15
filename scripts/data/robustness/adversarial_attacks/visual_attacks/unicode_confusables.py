import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.append(os.path.dirname(os.path.realpath(__file__)))

# Visual attacks with confusable characters based on:
# https://util.unicode.org/UnicodeJsps/confusables.jsp


class UnicodeConfusable:
    def __init__(self) -> None:
        self.dirname = os.path.dirname(os.path.realpath(__file__))
        self.confusables = self.load_json()
        self.punctuation = self.load_regular_punctuation()
        self.rng = np.random.default_rng(42)

    def load_json(self):
        file_path = os.path.join(self.dirname, "unicode_confusables.json")
        if not Path(file_path).is_file():
            loaded = self.get_confusables("unicode_confusables.txt")
            self.save_as_json(loaded, path=file_path)
        else:
            with open(file_path, "r", encoding="utf-8") as fp:
                loaded = json.load(fp)
        return loaded

    def get_confusables(self, file: str) -> None:
        """
        Creates dict of unicode confusables
        """

        def every_item_as_key(line: list) -> list:
            line = line[1:]  # ignore leading '#'
            return {key: [v for v in line if v != key] for key in line}

        file_path = os.path.join(self.dirname, file)
        results_dict = {}
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f.readlines():
                tok = line.strip().split("\t")
                if line[0] == "#" and len(tok) > 1:
                    results_dict.update(every_item_as_key(tok))
        return results_dict

    def save_as_json(self, results: dict, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=4, ensure_ascii=False)
        return

    def load_regular_punctuation(self):
        """
        Returns list of common punctuation
        """
        # gparent = Path(self.dirname).parents[1]
        fp = os.path.join(self.dirname, "punctuation.txt")
        store = []
        with open(fp, "r", encoding="utf-8") as f:
            for line in f.readlines():
                store.append(line.strip())
        return store

    def __call__(self, sentence: str, prob: float, spare_punctuation: bool = True) -> str:
        """
        Returns perturbed sentence.
        Args
            sentence: str to be perturbed
            prob: probability of perturbation
            spare_punctuation: whether to not perturb punctuation
        """
        out_str = []
        for c in sentence:
            if c == " " or (c in self.punctuation and spare_punctuation):
                s = c
            elif c not in self.confusables.keys():
                sys.stdout.write(f"Character '{c}' not in dict of confusables. ")
                s = c
            else:
                if self.rng.random() < prob:
                    s = self.rng.choice(self.confusables[c], size=1)[0]
                else:
                    s = c
            out_str.append(s)
        return "".join(out_str)
