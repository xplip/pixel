import random


def manip_segmentations(sentence, p, allow_remove=True, allow_insert=False):
    newsentence = ""
    for char in sentence:
        if allow_insert and random.random() < p / 10:
            newsentence += " "
        if not (allow_remove and char == " " and random.random() < p):
            newsentence += char
    return newsentence
