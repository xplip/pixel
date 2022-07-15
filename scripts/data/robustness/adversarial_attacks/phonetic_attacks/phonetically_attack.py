# Don't forget to cite https://github.com/letter-to-phoneme/m2m-aligner and https://github.com/cmusphinx/cmudict

import os
import pickle

import numpy as np


def softmax(x, theta=1):
    ps = np.exp(x * theta)
    ps /= np.sum(ps)
    return ps


class PhoneticAttacker:
    def __init__(self, stats_folder="statistics"):
        self.ngrams = {}
        with open(os.path.join(stats_folder, "gram_1.pkl"), "rb") as f:
            self.ngrams[1] = pickle.load(f)
        with open(os.path.join(stats_folder, "gram_2.pkl"), "rb") as f:
            self.ngrams[2] = pickle.load(f)
        with open(os.path.join(stats_folder, "gram_3.pkl"), "rb") as f:
            self.ngrams[3] = pickle.load(f)
        with open(os.path.join(stats_folder, "cmu_map.pkl"), "rb") as f:
            self.cmu_map = pickle.load(f)
        self.retrieve_map = {1: 0, 2: 1, 3: 1}

        # Hyperparams
        self.min_examples = 8  # At how few examples do we switch from 3-gram to 2-gram etc.
        self.softmax_param = 4  # High softmax_param => low randomness. Choose None to always choose maximum.
        self.word_attack_prob = 1
        self.phoneme_attack_prob = 1

    def set_level(self, p):
        # p between 0 and 1
        # Some empirically tuned way to select softmax param and word_attack_prob
        # to approximately match the severities of the other attacks
        self.softmax_param = 0.5 + (2 ** ((1 - p) * 5)) / 8
        self.word_attack_prob = self.phoneme_attack_prob = -(1 / (p * 20 + 1)) + 1

    def reconstruct_word(self, phonemes):
        word = []
        stats = {}
        for key in self.ngrams:
            stats[key] = self.calc_word_stats(phonemes, key)
        for i in range(0, len(stats[1])):
            for cur_n in range(3, 0, -1):
                phone_stats = stats[cur_n][i]
                sum_examples = sum(phone_stats.values())
                if sum_examples < self.min_examples and cur_n > 1:
                    continue
                letters, nums = zip(*list(phone_stats.items()))
                nums = np.array(nums) / sum_examples
                if self.softmax_param is None:
                    word.append(letters[np.argmax(nums)])
                else:
                    probs = softmax(nums, self.softmax_param)
                    word.append(np.random.choice(letters, p=probs))
                break
        return word

    def preprocess_sentence(self, sentence):
        in_word_chars = "abcdefghijklmnopqrstuvwxyz'"
        cap_letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        output = []
        capitatlize = []
        uptilnow = ""
        for char in sentence:
            if char in in_word_chars:
                uptilnow += char
            elif char in cap_letters:
                if len(uptilnow) == 0:
                    capitatlize.append(len(output))
                uptilnow += char.lower()
            else:
                if len(uptilnow) > 0:
                    output.append(uptilnow)
                output.append(char)
                uptilnow = ""
        if len(uptilnow) > 0:
            output.append(uptilnow)
        return output, capitatlize

    def __call__(self, sentence, prob):
        self.set_level(prob)
        tokens, capitalize = self.preprocess_sentence(sentence)
        out_sentence = []
        for token in tokens:
            if np.random.rand() > self.word_attack_prob:
                out_sentence.append(token)
                continue
            if token not in self.cmu_map:
                out_sentence.append(token)
                continue
            letters, phonemes = self.cmu_map[token]
            reconstructed = self.reconstruct_word(phonemes)
            new_word = ""
            for orig_letter, recon_letter in zip(letters[1:-1], reconstructed):
                if np.random.rand() > self.phoneme_attack_prob:
                    new_word += ":" if orig_letter == ":" else orig_letter.replace(":", "")
                else:
                    new_word += recon_letter.replace(":", "")
            out_sentence.append(new_word)
        for cap in capitalize:
            out_sentence[cap] = out_sentence[cap].capitalize()
        return "".join(out_sentence)

    def attack_document(self, filname, output):
        with open(filname, "r") as f:
            sentences = f.read().splitlines()
        out = []
        for sentence in sentences:
            out.append(self.__call_(sentence))
        with open(output, "w") as f:
            f.write("\n".join(["\t".join(x) for x in zip(sentences, out)]))

    def calc_word_stats(self, phonemes, n):
        ngram = self.ngrams[n]
        stats = []
        for i in range(1, len(phonemes) - 1):
            if n == 1:
                search = (phonemes[i],)
            elif n == 2:
                search = (phonemes[i - 1], phonemes[i])
            elif n == 3:
                search = (phonemes[i - 1], phonemes[i], phonemes[i + 1])
            retrieve = self.retrieve_map[n]
            letter_dict = ngram[search]
            out_dict = {}
            for key, value in letter_dict.items():
                if key[retrieve] in out_dict:
                    out_dict[key[retrieve]] += value
                else:
                    out_dict[key[retrieve]] = value
            stats.append(out_dict)
        return stats


if __name__ == "__main__":
    attacker = PhoneticAttacker()
    attacker.attack_document("sts-b-sentences.txt", "examples/lil_random_with_freq.txt")
