#!/usr/bin/env python

from collections import Counter
from math import log2
from typing import Iterable


class TfIdf:
    """
    Collects the term frequency (TF) and inverse document frequency (IDF) scores of a
    set of documents. All calls to `add_document()` should be done before calling the
    `__call__()` function.
    """

    def __init__(self):
        """
        Initialises the frequency collections.
        """
        self.term_frequencies: list[dict[str, float]] = []
        self.inverse_doc_frequencies: Counter = Counter()

    @property
    def nr_documents(self) -> int:
        """
        Returns the number of documents that have been added so far. This function
        relies on `term_frequencies` having a single element for every document.

        :return: The current number of documents.
        """
        return len(self.term_frequencies)

    def add_document(self, document: Iterable[str]) -> int:
        """
        Adds a document to the collection. The TF score is computed for each distinct
        term in the document immediately. The IDF score can't be computed yet, and thus
        only the occurring terms are added to the inverse document frequency mapping.

        :param document: The contents of the document as an iterable collection of
            words.
        :return: An identifier for the added document.
        """
        occurrences = Counter(document)
        most_common = occurrences.most_common(1)[0][1]
        self.term_frequencies.append(
            {term: count / most_common for term, count in occurrences.items()}
        )
        self.inverse_doc_frequencies += {term: 1 for term in occurrences}
        return len(self.term_frequencies) - 1

    def __call__(self, document_id: int, term: str) -> float:
        """
        Returns the TF.IDF score for the given term in the given document. This method
        should only be called once all of the documents have been added.

        :param document_id: The identifier of the document.
        :param term: The term for which the TF.IDF score is being requested.
        :return: The TF.IDF score.
        """
        if term not in self.term_frequencies[document_id]:
            return 0.0
        tf = self.term_frequencies[document_id][term]
        idf = log2(self.nr_documents / self.inverse_doc_frequencies[term])
        return tf * idf


if __name__ == "__main__":
    pass
