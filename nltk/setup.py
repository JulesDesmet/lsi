#!/usr/bin/env python

from os.path import dirname
from nltk import download

# https://www.geeksforgeeks.org/python-lemmatization-approaches-with-examples/
download("wordnet", dirname(__file__))
download("averaged_perceptron_tagger", dirname(__file__))
