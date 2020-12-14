#!/usr/bin/env python

from math import log2
from typing import Callable, Iterable, Union


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
        self.nr_documents: int = 0
        self.terms: list[str] = []
        self.term_ids: dict[str, int] = {}
        self.terms_per_doc: list[dict[int, float]] = []
        self.docs_per_term: list[set[int]] = []
        self.tfidf_scores: list[dict[int, float]] = []

    def __iadd__(self, tfidf: "TfIdf") -> "TfIdf":
        """
        Adds the results from a different TfIdf object to this object's results.

        :param tfidf: The other TfIdf object.
        :return: `self`
        """
        if self.tfidf_scores or tfidf.tfidf_scores:
            raise Exception("Optimised TfIdf objects can't be merged...")

        # Create a translation dictionary, which maps the other TfIdf object's term IDs
        # to this object's term IDs; also insert missing terms in this object
        translation = {}
        for term, term_id in tfidf.term_ids.items():
            if term not in self.term_ids:
                self.term_ids[term] = len(self.terms)
                self.terms.append(term)
                self.docs_per_term.append(tfidf.docs_per_term[term_id])
            else:
                self.docs_per_term[term_id] |= self.docs_per_term[term_id]
            translation[term_id] = self.term_ids[term]

        for document_freqs in tfidf.terms_per_doc:
            self.terms_per_doc.append(
                {translation[term_id]: freq for term_id, freq in document_freqs.items()}
            )

        self.nr_documents += tfidf.nr_documents
        return self

    def process_document(self, document: Iterable[str]) -> dict[str, float]:
        """
        Computes the TF scores of each term in the document.

        :return: A dictionary containing all of the document's terms and their TF
            scores.
        """
        occurrences = Counter(document)
        if not occurrences:
            return {}
        most_common = occurrences.most_common(1)[0][1]
        return {term: count / most_common for term, count in occurrences.items()}

    def add_document(self, document: Iterable[str]) -> int:
        """
        Adds a document to the collection. The TF score is computed for each distinct
        term in the document immediately. The IDF score can't be computed yet, and thus
        only the occurring terms are added to the inverse document frequency mapping.

        :param document: The contents of the document as an iterable collection of
            words.
        :return: An identifier for the added document, or -1 if the document wasn't
            added.
        """
        document_id = self.nr_documents
        term_frequencies = self.process_document(document)

        # Add the terms to the list if it's not yet included
        for term in term_frequencies:
            if term not in self.term_ids:
                self.term_ids[term] = len(self.terms)
                self.terms.append(term)

        # Create a new dictionary that maps the term IDs instead of the terms
        term_id_frequencies: dict[int, int] = {
            self.term_ids[term]: count for term, count in term_frequencies.items()
        }

        self.terms_per_doc.append(term_id_frequencies)
        for term_id in term_id_frequencies:
            if term_id >= len(self.docs_per_term):
                self.docs_per_term.extend(
                    set() for _ in range(term_id - len(self.docs_per_term))
                )
                self.docs_per_term.append({document_id})
            else:
                self.docs_per_term[term_id].add(document_id)

        self.nr_documents += 1
        return document_id

    def optimise(self) -> None:
        """
        Computes all of the TF.IDF scores from the `terms_per_doc` and `docs_per_term`
        variables. These are afterwards cleared to save some memory space.
        """
        self.tfidf_scores = [None for _ in range(self.nr_documents)]

        # Compute the TF.IDF scores for each document ID and term ID
        # Additionally, sort the terms for each document
        for document_id in range(self.nr_documents - 1, -1, -1):
            self.tfidf_scores[document_id] = {
                term: freq * log2(self.nr_documents / len(self.docs_per_term[term]))
                for term, freq in sorted(self.terms_per_doc[-1].items())
                if self.nr_documents != len(self.docs_per_term[term])
            }
            del self.terms_per_doc[-1]

    def call_pre_optimise(self, document_id: int, term_id: int) -> float:
        """
        Returns the TF.IDF score for the given term in the given document. This method
        should only be called once all of the documents have been added and before
        `optimise()` has been called.

        :param document_id: The identifier of the document.
        :param term_id: The term ID for which the TF.IDF score is being requested.
        :return: The TF.IDF score.
        """
        tf = self.terms_per_doc[document_id][term_id]
        idf = log2(self.nr_documents / len(self.docs_per_term[term_id]))
        return tf * idf

    def call_post_optimise(self, document_id: int, term_id: int) -> float:
        """
        Returns the TF.IDF score for the given term in the given document. This method
        should only be called after `optimise()` has been called.

        :param document_id: The identifier of the document.
        :param term_id: The term ID for which the TF.IDF score is being requested.
        :return: The TF.IDF score.
        """
        return self.tfidf_scores[document_id][term_id]

    def __call__(self, document_id: int, term: Union[str, int]) -> float:
        """
        Returns the TF.IDF score for the given term in the given document. This method
        should only be called once all of the documents have been added.

        :param document_id: The identifier of the document.
        :param term: The term or term ID for which the TF.IDF score is being requested.
        :return: The TF.IDF score.
        """
        if isinstance(term, str) and term not in self.term_ids:
            return 0.0
        term_id = term if isinstance(term, int) else self.term_ids[term]

        # Because `optimise()` is the only function adding to `tfidf_scores`, we can use
        # it to find out whether this function has been called already. If it has been
        # called but `tfidf_scores` is still empty, then it shouldn't matter which
        # function is picked.
        if self.tfidf_scores:
            if term_id not in self.tfidf_scores[document_id]:
                return 0.0
            return self.call_post_optimise(document_id, term_id)
        else:
            if term_id not in self.terms_per_doc[document_id]:
                return 0.0
            return self.call_pre_optimise(document_id, term_id)

    def __matmul__(
        self, vector: Iterable[float], term_major: bool = None
    ) -> list[float]:
        """
        Multiplies the term-document-matrix, or its transpose, with a vector.

        :param vector: A vector (i.e. an (n x 1)-matrix) whose dimension n matches one
            of the sizes of the term-document-matrix, depending on the parameter
            `term_major`. If `term_major` is True, then this vector should have the same
            size as the matrix' number of documents. If it's `False`, then it should
            match the matrix' number of terms. If it's not specified (or `None`), then
            the size can be either.
        :param term_major: Whether the vector is to be multiplied with the
            term-document-matrix or its transpose. `True` means that the rows are
            associated with a term each, whereas `False` means that the rows are
            associated with documents. The columns are associated with documents or
            terms respectively.
            If this parameter isn't specified, then the length of the vector will
            determine which matrix it's multiplied with. If the number of documents is
            equal to the number of terms, then this parameter is simply assumed to be
            `True`.
        :return: The product of the matrix and the vector, as a list of `float`s.
        """
        # If neither term major nor document major was selected, determine based on
        # vector length
        if term_major is None:
            term_major = len(vector) == len(self.terms)

        if term_major:
            assert len(vector) == self.nr_documents
            return [
                sum(
                    self.tfidf_scores[document_id][term_id] * vector[document_id]
                    for document_id in documents
                )
                for term_id, documents in enumerate(self.docs_per_term)
            ]
        else:
            assert len(vector) == len(self.terms)
            return [
                sum(
                    score * vector[term_id]
                    for term_id, score in self.terms_per_doc[document_id].items()
                )
                for document_id in range(self.nr_documents)
            ]
