#!/usr/bin/env python

import csv
from os.path import dirname, join
from typing import Generator, Iterable

from nltk import data, pos_tag_sents, sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet


# This should be the directory root (assuming this file is in `{product_root}/src/`)
data.path = [join(dirname(dirname(__file__)), "nltk")]


def read_csv(filename: str) -> Generator[dict[str, str], None, None]:
    """
    Reads a CSV file line by line. This method should be used as follows:
    ```
    for row in read_csv(...):
        do_something_with_row(...)
    ```

    :param filename: The CSV file, relative from the directory from where the script is
        being run, unless it's an absolute path.
    :return: A generator that yields the rows of the CSV file as a dictionary of fields.
    """
    with open(filename) as csv_file:
        reader = csv.reader(csv_file)
        fields = next(reader)
        # dict(zip(a, b)) creates a dictionary with keys from a and values from b
        yield from (dict(zip(fields, row)) for row in reader)


def split_text(text: str) -> list[list[str]]:
    """
    Splits the text strings into a list of sentences, which are lists of words.

    :param text: A string of text.
    :return: A list of lists of strings, which contain the text per word per sentence.
    """
    return [word_tokenize(sentence) for sentence in sent_tokenize(text)]


def lemmatize(text: Iterable[Iterable[str]]) -> Generator[str, None, None]:
    """
    Replaces each word in the text string with its lemma. This method uses the nltk
    library, and requires some NLTK data in order to run. This data can be downloaded
    by running `setup.py` in the `nltk` directory.

    (https://www.geeksforgeeks.org/python-lemmatization-approaches-with-examples/)

    :param text: The text that is to be lemmatised as a sequence of sentences, which are
        sequences of words. These words/strings should only contain lowercase letters.
    :return: A generator that yields the lemmas. These can be concatenated into a string
        by using the " ".join(...) function.
    """
    tag_dict = {
        "J": wordnet.ADJ,
        "N": wordnet.NOUN,
        "R": wordnet.ADV,
        "V": wordnet.VERB,
    }
    lemmatizer = WordNetLemmatizer()
    # Tags the words with their type/pos (part of speech)
    # Then translates the tags using `tag_dict`
    # And finally lemmatizes the words if possible
    return (
        lemmatizer.lemmatize(word, tag_dict[tag[0]]) if tag[0] in tag_dict else word
        for sentence in pos_tag_sents(text)  # Outer loop
        for word, tag in sentence  # Inner loop
    )


if __name__ == "__main__":

    for index, data in enumerate(read_csv("data/news_dataset.csv")):
        text = split_text(data["content"])

        print(" ".join(lemmatize(text))[:200])

        if index == 10:
            break
