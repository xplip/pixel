import os
from collections import defaultdict


class PerturbationsStorage(object):
    def __init__(self, perturbations_file_path):
        self.perturbations_file_path = perturbations_file_path
        self.observed_perturbations = defaultdict(lambda: set())
        if os.path.exists(self.perturbations_file_path):
            self.read()

    def read(self):
        with open(self.perturbations_file_path, "r") as f:
            for l in f:
                key, values = l.strip().split("\t")
                values = values.split()
                self.observed_perturbations[key] |= set(values)

    def maybe_write(self):
        if self.perturbations_file_path:
            with open(self.perturbations_file_path, "w") as f:
                for k, v in self.observed_perturbations.items():
                    f.write("{}\t{}\n".format(k, " ".join(v)))

    def add(self, key, value):
        self.observed_perturbations[key].add(value)

    def observed(self, key, value):
        return value in self.observed_perturbations[key]
