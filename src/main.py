#!/usr/bin/env python

from argparse import ArgumentParser, Namespace
from logging import debug, DEBUG, getLogger
from multiprocessing import Process, Queue, set_start_method
from queue import Empty
from time import time
from typing import Optional

from preprocessing import read_csv, split_text, lemmatize, remove_stopwords
from term_doc_matrix import TfIdf
from svd import SVD

class BaseProcess:
    """
    A base class for the `WorkerProcess` and `ManagerProcess` classes. It provides the
    functionality to preprocess and process the input data.
    """

    # The TF.IDF collection object for the process
    tfidf: TfIdf
    # A dictionary mapping the IDs of the TF.IDF objects to their indices
    data_ids: dict[int, int]

    def __init__(self):
        """
        Initialises the TF.IDF collection and the data IDs dictionary.
        """
        self.tfidf = TfIdf()
        self.data_ids = {}

    @staticmethod
    def get_data(document: dict[str, str]) -> tuple[int, str]:
        """
        Returns the ID and the content of the document.

        :param document: A single document as a dictionary of attributes.
        :return: The document ID and content, as a tuple.
        """
        return int(document["id"]), document["title"] + " " + document["content"]

    def process_data(
        self, data_id: int, data: str, add_document: bool
    ) -> Optional[dict[str, float]]:
        """
        Preprocesses and processes a document.

        :param data_id: The document's ID.
        :param data: The content of the document.
        :param add_document: Whether the document should immediately be added to the
            TF.IDF collection.
        :return: The TF.IDF scores of each term in the document, unless the document was
            added to the collection. In that case nothing is returned.
        """
        preprocessed = remove_stopwords(lemmatize(split_text(data)))
        if not add_document:
            return self.tfidf.process_document(preprocessed)
        else:
            index = self.tfidf.add_document(preprocessed)
            self.data_ids[data_id] = index

    def run(self, filename: str) -> None:
        """
        Processes a collection of documents in a from a CSV file.

        :param filename: The name of the CSV file; relative to the directory from where
            the code is being run.
        """
        for data in read_csv(filename):
            self.process_data(*self.get_data(data), True)
        self.tfidf.optimise()


class WorkerProcess(BaseProcess):
    """
    A worker process that processes documents it receives in a queue. This should be
    used with a `ManagerProcess`, which sends documents through queues to its workers.
    """

    def __init__(self, document_queue: Queue, result_queue: Queue, proc_id: int):
        """
        Initialises the `Process` object and the document queue.

        :param queue: The queue through which the worker receives the documents it's
            supposed to be processing.
        """
        super().__init__()
        self.document_queue = document_queue
        self.result_queue = result_queue
        self.process = Process(target=self.run)
        self.proc_id = proc_id

    def run(self) -> None:
        """
        Processes all of the documents it receives in its queue. The process will stop
        once it finds `None` in its queue.
        """
        try:
            debug(f"Process {self.proc_id} has started.")
            while (data := self.document_queue.get()) is not None:
                self.process_data(*data, True)
            self.result_queue.put((self.tfidf, self.data_ids))
            debug(f"Process {self.proc_id} has finished.")

        except KeyboardInterrupt:
            debug(f"Process {self.proc_id} has been interrupted.")


class ManagerProcess(BaseProcess):
    """
    A manager process that uses worker processes to process a collection of documents.
    """

    def __init__(self, nr_procs: int = 1):
        """
        Initialises the worker processes and their queues.

        :param nr_procs: The number of processes that are to be used in total, including
            the manager.
        """
        super().__init__()
        self.document_queue = Queue()
        self.result_queue = Queue()
        self.processes = [
            WorkerProcess(self.document_queue, self.result_queue, i + 1)
            for i in range(nr_procs - 1)
        ]

    def run(self, filename: str) -> None:
        """
        Processes a CSV file of documents using a number of worker processes.

        :param filename: The name of the CSV file; relative to the directory from where
            the code is being run.
        """
        try:
            # Start the worker processes
            for proc in self.processes:
                proc.process.start()
            nr_procs = len(self.processes) + 1

            # Simply call the runner of the `BaseProcess` subclass if there are no workers
            if nr_procs == 1:
                BaseProcess.run(self, filename)
                return

            # Distribute the documents by simply putting them in a queue
            debug("Process 0 has started.")
            for data in read_csv(filename):

                if self.document_queue.qsize() < 3 * nr_procs:
                    self.document_queue.put(self.get_data(data))
                else:
                    self.process_data(*self.get_data(data), True)
            debug("Process 0 has finished.")

            # Signal to the workers that all of the documents have been distributed
            for _ in range(nr_procs - 1):
                self.document_queue.put(None)

            # Collect all of the TF scores returned by the workers
            debug("Result aggregation has started.")
            for _ in range(nr_procs - 1):
                tfidf, data_ids = self.result_queue.get()

                self.tfidf += tfidf
                offset = len(self.data_ids)
                for data_id, document_id in data_ids.items():
                    self.data_ids[data_id] = offset + document_id

            self.tfidf.optimise()
            debug("Result aggregation has finished.")

            # Wait for all of the workers to finish
            for proc in self.processes:
                proc.process.join()

        except KeyboardInterrupt:
            self.document_queue.close()
            self.result_queue.close()
            debug("Process 0 has been interrupted.")


def parse_arguments() -> Namespace:
    """
    Parses the command line arguments using the `argparse` library.

    :return: A namespace containing the parsed command line arguments.
    """
    parser = ArgumentParser()
    parser.add_argument("filename", nargs="?", default="data/news_dataset.csv")
    parser.add_argument("--threads", "-j", default=1, type=int)

    return parser.parse_args()


if __name__ == "__main__":
    getLogger().setLevel(DEBUG)
    args = parse_arguments()

    start = time()
    if args.threads == 1:
        process = BaseProcess()
        process.run(args.filename)
    else:
        manager = ManagerProcess(args.threads)
        manager.run(args.filename)


    decomposition = SVD(manager.tfidf)

    decomposition.matrix_multiplication()
    end = time()
    t = end - start
    debug(f"{int(t) // 60} minutes {t % 60} seconds")
