#!/usr/bin/env python

from os.path import abspath, join
from nltk import download


# https://www.geeksforgeeks.org/python-lemmatization-approaches-with-examples/
download("wordnet", join(abspath("."), "nltk"))
download("averaged_perceptron_tagger", join(abspath("."), "nltk"))
