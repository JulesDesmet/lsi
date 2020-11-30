#!/usr/bin/env python

from os.path import dirname
from nltk import download

# https://www.geeksforgeeks.org/python-lemmatization-approaches-with-examples/

if __name__ != "__main__":
    raise ImportError("Don't import this module!")

directory = dirname(__file__)
download("wordnet", directory)
download("averaged_perceptron_tagger", directory)
download("punkt", directory)
