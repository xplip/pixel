import os
import pickle

from tqdm import tqdm


def insert_to_ngrams(ngram, phonemes, letters, amount=1):
    if phonemes in ngram:
        if letters in ngram[phonemes]:
            ngram[phonemes][letters] += amount
        else:
            ngram[phonemes][letters] = amount
    else:
        ngram[phonemes] = {letters: amount}


def calc_frequencies(infile="word_frequency.txt", out_file="statistics/word_frequency.pkl"):
    word_freq_dict = {}
    with open(infile, "r") as f:
        for line in f.read().splitlines():
            word, freq = line.split(" ")
            word_freq_dict[word] = int(freq)
    with open(out_file, "wb") as f:
        pickle.dump(word_freq_dict, f)
    return word_freq_dict


def calc_statistics(infile="cmudict.aligned", out_folder="statistics", word_frequencies=None):
    gram_1 = {}
    gram_2 = {}
    gram_3 = {}
    cmu_mapping = {}
    with open(infile, "r") as f:
        lines = f.read().splitlines()
    for line in tqdm(lines):
        letters, phonemes = [["[START]"] + x.split("|")[:-1] + ["[END]"] for x in line.split("\t")]
        orig_word = "".join(letters[1:-1]).replace(":", "")
        cmu_mapping[orig_word] = (tuple(letters), tuple(phonemes))
        if word_frequencies is None or orig_word not in word_frequencies:
            amount = 1
        else:
            amount = word_frequencies[orig_word]
        for i in range(1, len(letters) - 1):
            insert_to_ngrams(gram_1, (phonemes[i],), (letters[i],), amount)
            insert_to_ngrams(gram_2, (phonemes[i - 1], phonemes[i]), (letters[i - i], letters[i]), amount)
            insert_to_ngrams(
                gram_3,
                (phonemes[i - 1], phonemes[i], phonemes[i + 1]),
                (letters[i - i], letters[i], letters[i + 1]),
                amount,
            )
    with open(os.path.join(out_folder, "gram_1.pkl"), "wb") as f:
        pickle.dump(gram_1, f)
    with open(os.path.join(out_folder, "gram_2.pkl"), "wb") as f:
        pickle.dump(gram_2, f)
    with open(os.path.join(out_folder, "gram_3.pkl"), "wb") as f:
        pickle.dump(gram_3, f)
    with open(os.path.join(out_folder, "cmu_map.pkl"), "wb") as f:
        pickle.dump(cmu_mapping, f)


if __name__ == "__main__":
    with open("statistics/word_frequency.pkl", "rb") as f:
        word_frequencies = pickle.load(f)
    calc_statistics(word_frequencies=word_frequencies)
