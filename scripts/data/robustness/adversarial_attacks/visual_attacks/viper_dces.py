"""
Source https://github.com/UKPLab/naacl2019-like-humans-visual-attacks/blob/master/code/VIPER/viper_dces.py
(Modified)
"""

import argparse
import os
import random
import sys

sys.path.append(os.path.dirname(os.path.realpath(__file__)))

import numpy as np
import pandas as pd
from perturbations_store import PerturbationsStorage
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import NearestNeighbors


def read_data_conll(fn):

    docs = []
    doc = []

    for line in open(fn):
        line = line.strip()
        if line == "":
            if doc != []:
                a, b = [], []
                for l in doc:
                    _, x, y = l.split("\t")
                    a.append(x)
                    b.append(y)
                docs.append("{}\t{}".format(" ".join(a), " ".join(b)))
            doc = []
        else:
            doc.append(line)

    if doc != []:
        a, b = [], []
        for l in doc:
            _, x, y = l.split("\t")
            a.append(x)
            b.append(y)
        docs.append("{}\t{}".format(" ".join(a), " ".join(b)))

    return docs


def read_data_standard(fn):
    docs = []
    for line in open(fn):
        docs.append(line)
    return docs


def char_to_hex_string(ch):
    return "{:04x}".format(ord(ch)).upper()


# function for retrieving the variations of a character


class ViperDCES:
    def __init__(self):
        # load the unicode descriptions into a single dataframe with the chars as indices
        self.dirname = os.path.dirname(os.path.realpath(__file__))
        self.descs = pd.read_csv(
            os.path.join(self.dirname, "NamesList.txt"),
            skiprows=np.arange(16),
            header=None,
            names=["code", "description"],
            delimiter="\t",
        )
        self.descs = self.descs.dropna(0)
        self.descs_arr = self.descs.values  # remove the rows after the descriptions
        self.vectorizer = CountVectorizer(max_features=1000)
        self.desc_vecs = self.vectorizer.fit_transform(self.descs_arr[:, 0]).astype(float)
        self.vecsize = self.desc_vecs.shape[1]
        self.vec_colnames = np.arange(self.vecsize)
        self.desc_vecs = pd.DataFrame(self.desc_vecs.todense(), index=self.descs.index, columns=self.vec_colnames)
        self.descs = pd.concat([self.descs, self.desc_vecs], axis=1)
        self.disallowed = [
            "TAG",
            "MALAYALAM",
            "BAMUM",
            "HIRAGANA",
            "RUNIC",
            "TAI",
            "SUNDANESE",
            "BATAK",
            "LEPCHA",
            "CHAM",
            "TELUGU",
            "DEVANGARAI",
            "BUGINESE",
            "MYANMAR",
            "LINEAR",
            "SYLOTI",
            "PHAGS-PA",
            "CHEROKEE",
            "CANADIAN",
            "YI",
            "LYCIAN",
            "HANGUL",
            "KATAKANA",
            "JAVANESE",
            "ARABIC",
            "KANNADA",
            "BUHID",
            "TAGBANWA",
            "DESERET",
            "REJANG",
            "BOPOMOFO",
            "PERMIC",
            "OSAGE",
            "TAGALOG",
            "MEETEI",
            "CARIAN",
            "UGARITIC",
            "ORIYA",
            "ELBASAN",
            "CYPRIOT",
            "HANUNOO",
            "GUJARATI",
            "LYDIAN",
            "MONGOLIAN",
            "AVESTAN",
            "MEROITIC",
            "KHAROSHTHI",
            "HUNGARIAN",
            "KHUDAWADI",
            "ETHIOPIC",
            "PERSIAN",
            "OSMANYA",
            "ELBASAN",
            "TIBETAN",
            "BENGALI",
            "TURKIC",
            "THROWING",
            "HANIFI",
            "BRAHMI",
            "KAITHI",
            "LIMBU",
            "LAO",
            "CHAKMA",
            "DEVANAGARI",
            "ITALIC",
            "CJK",
            "MEDEFAIDRIN",
            "DIAMOND",
            "SAURASHTRA",
            "ADLAM",
            "DUPLOYAN",
        ]

        self.disallowed_codes = ["1F1A4", "A7AF"]

    def get_all_variations(self, ch):

        # get unicode number for c
        c = char_to_hex_string(ch)

        # problem: latin small characters seem to be missing?
        if np.any(self.descs["code"] == c):
            description = self.descs["description"][self.descs["code"] == c].values[0]
        else:
            print("Failed to disturb %s, with code %s" % (ch, c))
            return c, np.array([])

        # strip away everything that is generic wording, e.g. all words with > 1 character in
        toks = description.split(" ")

        case = "unknown"

        identifiers = []
        for tok in toks:

            if len(tok) == 1:
                identifiers.append(tok)

                # for debugging
                if len(identifiers) > 1:
                    print("Found multiple ids: ")
                    print(identifiers)

            elif tok == "SMALL":
                case = "SMALL"
            elif tok == "CAPITAL":
                case = "CAPITAL"

        # for debugging
        # if case == 'unknown':
        #    sys.stderr.write('Unknown case:')
        #    sys.stderr.write("{}\n".format(toks))

        # find matching chars
        matches = []

        for i in identifiers:
            for idx in self.descs.index:
                desc_toks = self.descs["description"][idx].split(" ")
                if (
                    i in desc_toks
                    and not np.any(np.in1d(desc_toks, self.disallowed))
                    and not np.any(np.in1d(self.descs["code"][idx], self.disallowed_codes))
                    and not int(self.descs["code"][idx], 16) > 30000
                ):

                    # get the first case descriptor in the description
                    desc_toks = np.array(desc_toks)
                    case_descriptor = desc_toks[(desc_toks == "SMALL") | (desc_toks == "CAPITAL")]

                    if len(case_descriptor) > 1:
                        case_descriptor = case_descriptor[0]
                    elif len(case_descriptor) == 0:
                        case = "unknown"

                    if case == "unknown" or case == case_descriptor:
                        matches.append(idx)

        # check the capitalisation of the chars
        return c, np.array(matches)

    # function for finding the nearest neighbours of a given word
    def get_unicode_desc_nn(self, c, topn=1):
        # we need to consider only variations of the same letter -- get those first, then apply NN
        c, matches = self.get_all_variations(c)

        if not len(matches):
            return [], []  # cannot disturb this one

        # get their description vectors
        match_vecs = self.descs[self.vec_colnames].loc[matches]

        # find nearest neighbours
        neigh = NearestNeighbors(metric="euclidean")
        Y = match_vecs.values
        neigh.fit(Y)

        X = self.descs[self.vec_colnames].values[self.descs["code"] == c]

        if Y.shape[0] > topn:
            dists, idxs = neigh.kneighbors(X, topn, return_distance=True)
        else:
            dists, idxs = neigh.kneighbors(X, Y.shape[0], return_distance=True)

        # turn distances to some heuristic probabilities
        # print(dists.flatten())
        probs = np.exp(-0.5 * dists.flatten())
        probs = probs / np.sum(probs)

        # turn idxs back to chars
        # print(idxs.flatten())
        charcodes = self.descs["code"][matches[idxs.flatten()]]

        # print(charcodes.values.flatten())

        chars = []
        for charcode in charcodes:
            chars.append(chr(int(charcode, 16)))

        # filter chars to ensure OOV scenario (if perturbations file from prev. perturbation contains any data...)
        c_orig = chr(int(c, 16))

        # print(chars)

        return chars, probs

    def __call__(self, sentence, prob):
        # the main loop for disturbing the text
        topn = 20

        mydict = {}
        out_x = []
        for c in sentence:
            # print(c)
            if c not in mydict:
                similar_chars, probs = self.get_unicode_desc_nn(c, topn=topn)
                probs = probs[: len(similar_chars)]

                # normalise after selecting a subset of similar characters
                probs = probs / np.sum(probs)

                mydict[c] = (similar_chars, probs)

            else:
                similar_chars, probs = mydict[c]

            r = random.random()
            if r < prob and len(similar_chars):
                s = np.random.choice(similar_chars, 1, replace=True, p=probs)[0]
            else:
                s = c
            out_x.append(s)
        return "".join(out_x)


if __name__ == "__main__":
    vdecs = ViperDCES()
    print(vdecs(0.5, sys.argv[1]))
