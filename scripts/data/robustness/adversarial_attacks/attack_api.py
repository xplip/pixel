import os
import sys

basepath = os.path.dirname(os.path.realpath(__file__))
sys.path.append(basepath)
import random

from phonetic_attacks.phonetically_attack import PhoneticAttacker
from simple_attacks.segmentations import manip_segmentations
from simple_attacks.simple_attacks import simple_perturb
from visual_attacks.unicode_confusables import UnicodeConfusable


class AdversarialAttacker:
    def __init__(self):
        self.phonetic_attacker = PhoneticAttacker(
            stats_folder=os.path.join(os.path.realpath(os.path.dirname(__file__)), "phonetic_attacks/statistics")
        )
        self.confusable_attacker = UnicodeConfusable()
        self.methods = [
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

    def do_one_attack(self, sentence, method, severity):
        if method not in self.methods:
            raise ValueError("Invalid method")
        elif method == "phonetic":
            return self.phonetic_attacker(sentence, severity)
        elif method == "confusable":
            return self.confusable_attacker(sentence, severity)
        elif method == "segmentation":
            return manip_segmentations(sentence, severity)
        else:
            return simple_perturb(sentence, method, severity)

    def multiattack(self, sentence, attacks_with_severity):
        """Perform multiple consecutive attacks
        :param sentence: Sentence to attack
        :param attacks_with_severity: List of tuples containing method and severity
        :return: The attacked sentence
        """
        for i, (attack, severity) in enumerate(attacks_with_severity):
            if attack == "rand":
                attack = random.choice(self.methods)
                while attack == "intrude" and i != len(attacks_with_severity) - 1:
                    attack = random.choice(self.methods)
            sentence = self.do_one_attack(sentence, attack, severity)
        return sentence


if __name__ == "__main__":
    attack = AdversarialAttacker()
    attacks_with_severity = [("confusable", 1)]
    print(attack.multiattack(sys.argv[1], attacks_with_severity))
