#!/usr/bin/env python

from preprocessing import read_csv, split_text, lemmatize, remove_stopwords
from term_doc_matrix import TfIdf


if __name__ == "__main__":

    # Allow some arguments for this script (in this exact order without leaving any out)
    # 1. A dataset in CSV format            (default = data/news_dataset.csv)
    # 2. A column header of the dataset     (default = content)
    from sys import argv
    filename = argv[1] if len(argv) >= 2 else "data/news_dataset.csv"
    col_name = argv[2] if len(argv) >= 3 else "content"

    tfidf = TfIdf()
    data_ids = {}

    for index, data in enumerate(read_csv(filename)):
        text = split_text(data[col_name])
        words = remove_stopwords(lemmatize(text))

        data_ids["id"] = tfidf.add_document(words)

        if index == 999:
            break

    for term in tfidf.term_frequencies[0]:
        print(f"{term:20}{tfidf(0, term)}")
