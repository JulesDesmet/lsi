#!/usr/bin/env python

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
        return list(
            "".join(
                word if isinstance(word, str) else (word[0] * word[1])
                for word in document
            )
        )


class UtilityTestCase(BaseTestCase):
    """
    A test case that tests the utility function(s) of the `BaseTestCase` class.
    """

    def test_convert_document(self) -> None:
        """
        Tests the `BaseTestCase.convert_document()` function.
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
        Tests the `TfIdf.process_document()` function.
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


if __name__ == "__main__":
    main(verbosity=2)
