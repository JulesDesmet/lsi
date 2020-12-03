#!/usr/bin/env python

from collections import Counter
from math import log2
from typing import Iterable


class TfIdf:
    """"""

    term_frequencies: list[dict[str, float]]
    inverse_doc_frequencies: Counter

    def __init__(self):
        """"""
        self.term_frequencies = []
        self.inverse_doc_frequencies = Counter()

    @property
    def nr_documents(self) -> int:
        """"""
        return len(self.term_frequencies)

    def add_document(self, document: Iterable[str]) -> int:
        """"""
        occurrences = Counter(document)
        most_common = occurrences.most_common(1)[0][1]
        self.term_frequencies.append(
            {term: count / most_common for term, count in occurrences.items()}
        )
        self.inverse_doc_frequencies += {term: 1 for term in occurrences}
        return len(self.term_frequencies) - 1

    def __call__(self, document_id: int, term: str) -> float:
        """"""
        if term not in self.term_frequencies[document_id]:
            return 0.0
        tf = self.term_frequencies[document_id][term]
        idf = log2(self.nr_documents / self.inverse_doc_frequencies[term])
        return tf * idf


if __name__ == "__main__":
    pass
