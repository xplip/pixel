#!/usr/bin/env python

# This source code is licensed under the Apache-2.0 license found in the
# LICENSE file in the root directory of this source tree.

from setuptools import find_packages, setup

setup(
    name="pixel",
    version="0.0.1",
    author="Team PIXEL",
    author_email="p.rust@di.ku.dk",
    url="https://github.com/xplip/pixel",
    description="Research code for the paper 'Language Modelling with Pixels'",
    license="Apache",
    package_dir={"": "src"},
    packages=find_packages("pixel"),
    zip_safe=True,
)
