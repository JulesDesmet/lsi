#!/usr/bin/env python

from collections import Counter
from itertools import product
from math import log2
from typing import Union
from unittest import main, TestCase

from term_doc_matrix import TfIdf


class BaseTestCase(TestCase):
    """
    A base class for all of the test cases, with some utility functions.
    """

    @staticmethod
    def convert_document(document: list[Union[str, tuple[str, int]]]) -> list[str]:
        """
        Converts a document to a simple list of strings. This method replaces each tuple
        of a string and an integer to that string that number of times. For example, an
        input `["a", ("b", 2), "a"]` would be converted to `["a", "b", "b", "a"]`.

        :param document: The encoded document as a list of strings and tuples.
        :return: The decoded document as a list of strings.
        """
        return [
            word
            for code in document
            for word in ([code] if isinstance(code, str) else [code[0]] * code[1])
        ]


class UtilityTestCase(BaseTestCase):
    """
    A test case that tests the utility function(s) of the `BaseTestCase` class.
    """

    def test_convert_document(self) -> None:
        """
        Tests the `BaseTestCase.convert_document()` function.                  .
        """
        # Each tuple contains an input for `BaseTestCase.convert_document()` and the
        # expected output
        documents = (
            (["a", "b", "c"], ["a", "b", "c"]),
            (["a", ("b", 2), "a"], ["a", "b", "b", "a"]),
            ([("a", 10), ("b", 10)], ["a"] * 10 + ["b"] * 10),
            ([], []),
            ([("a", 1), ("b", 1)], ["a", "b"]),
            ([("a", 0)], []),
        )
        for document, expected in documents:
            with self.subTest(document=document, expected=expected):
                self.assertEqual(self.convert_document(document), expected)


class TfIdfTestCase(BaseTestCase):
    """"""

    def setUp(self) -> None:
        """
        Sets up the environment for a test, by initialising the TfIdf object. This
        method is run before every single test.
        """
        self.tfidf = TfIdf()

    def test_process_document(self) -> None:
        """
        Tests the `TfIdf.process_document()` function.                         .
        """
        documents = (
            (
                ([("a", 20), ("b", 10), ("c", 10), ("d", 10)]),
                {"a": 1.0, "b": 0.5, "c": 0.5, "d": 0.5},
            ),
            (["a"], {"a": 1.0}),
            ([], {}),
            (
                ["a", "b", "c", "d", "e"],
                {"a": 1.0, "b": 1.0, "c": 1.0, "d": 1.0, "e": 1.0},
            ),
        )
        for document, expected in documents:
            document = self.convert_document(document)

            with self.subTest(document=document, expected=expected):
                tf_scores = self.tfidf.process_document(document)
                self.assertEqual(tf_scores, expected)

    def test_add_document(self) -> None:
        """
        Tests the `TfIdf.add_document()` function.                             .
        """
        documents = (
            (
                ([("a", 20), ("b", 10), ("c", 10), ("d", 10)]),
                {"a": 1.0, "b": 0.5, "c": 0.5, "d": 0.5},
            ),
            (["a"], {"a": 1.0}),
            ([], {}),
            (
                ["a", "b", "c", "d", "e"],
                {"a": 1.0, "b": 1.0, "c": 1.0, "d": 1.0, "e": 1.0},
            ),
        )
        for document, expected in documents:
            document = self.convert_document(document)

            with self.subTest(document=document, expected=expected):
                document_id = self.tfidf.add_document(document)
                self.assertEqual(self.tfidf.term_frequencies[document_id], expected)

        self.assertEqual(
            self.tfidf.inverse_doc_frequencies, Counter(a=3, b=2, c=2, d=2, e=1)
        )

    def test_call(self) -> None:
        """
        Tests the `TfIdf.__call__()` function.                                 .
        """
        documents = [
            [("a", 10), ("b", 5), ("c", 5), "d"],
            ["b", "d"],
            [("a", 20), ("c", 20), "d"],
            ["a", ("c", 5), "d"],
        ]
        tf_scores = [
            {"a": 1.0, "b": 0.5, "c": 0.5, "d": 0.1},
            {"a": 0.0, "b": 1.0, "c": 0.0, "d": 1.0},
            {"a": 1.0, "b": 0.0, "c": 1.0, "d": 0.05},
            {"a": 0.2, "b": 0.0, "c": 1.0, "d": 0.2},
        ]
        idf_scores = {"a": log2(4 / 3), "b": log2(4 / 2), "c": log2(4 / 3), "d": 0}

        for document in documents:
            self.tfidf.add_document(self.convert_document(document))

        # Verify that the internal data of `self.tfidf` is correct
        self.assertEqual(
            self.tfidf.term_frequencies,
            # Filter out the 0.0 values because the `TfIdf` class does this as well
            [{term: tf for term, tf in doc.items() if tf != 0} for doc in tf_scores],
        )
        self.assertEqual(
            self.tfidf.inverse_doc_frequencies, Counter(a=3, b=2, c=3, d=4)
        )

        # Iterate over each document and term
        for document_id, term in product(range(len(documents)), idf_scores):
            with self.subTest(document_id=document_id, word=term):
                tfidf_score = self.tfidf(document_id, term)
                expected = tf_scores[document_id][term] * idf_scores[term]
                self.assertEqual(tfidf_score, expected)

        with self.subTest(document_id=4, word="a"):
            self.assertEqual(self.tfidf(4, "a"), 0)


if __name__ == "__main__":
    main(verbosity=2)
