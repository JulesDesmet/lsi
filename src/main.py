#!/usr/bin/env python

from argparse import ArgumentParser, Namespace
from multiprocessing import Process, Queue, set_start_method
from time import time
from typing import Any

from preprocessing import read_csv, split_text, lemmatize, remove_stopwords
from term_doc_matrix import TfIdf


class BaseProcess:
    """
    A base class for the `WorkerProcess` and `ManagerProcess` classes. It provides the
    functionality to preprocess and process the input data.
    """

    # The TF.IDF collection object for the process
    tfidf: TfIdf
    # A dictionary mapping the IDs of the TF.IDF object to the document IDs
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

    def process_data(self, data_id: int, data: str) -> None:
        """
        Preprocesses and processes a document.

        :param data_id: The document's ID.
        :param data: The content of the document.
        """
        preprocessed = remove_stopwords(lemmatize(split_text(data)))
        index = self.tfidf.add_document(preprocessed)
        if index != -1:
            self.data_ids[index] = data_id

    def run(self, filename: str) -> None:
        """
        Processes a collection of documents in a from a CSV file.

        :param filename: The name of the CSV file; relative to the directory from where
            the code is being run.
        """
        for data in read_csv(filename):
            self.process_data(*self.get_data(data))


class WorkerProcess(BaseProcess):
    """
    A worker process that processes documents it receives in a queue. This should be
    used with a `ManagerProcess`, which sends documents through queues to its workers.
    """

    process: Process
    queue: Queue

    def __init__(self, queue: Queue):
        """
        Initialises the `Process` object and the document queue.

        :param queue: The queue through which the worker receives the documents it's
            supposed to be processing.
        """
        super().__init__()
        self.queue = queue
        self.process = Process(target=self.run)

    def run(self) -> None:
        """
        Processes all of the documents it receives in its queue. The process will stop
        once it finds `None` in its queue.
        """
        while (data := self.queue.get()) is not None:
            self.process_data(*data)


class ManagerProcess(BaseProcess):
    """
    A manager process that uses worker processes to process a collection of documents.
    """

    # The queue through which documents are sent to the workers
    queue: Queue
    # The worker processes this manager manages
    processes: list[WorkerProcess]
    # Whether the manager also processes documents; otherwise it just distributes them
    worker_manager: bool

    def __init__(self, nr_procs: int = 1, worker_manager: bool = False):
        """
        Initialises the worker processes and their queues.

        :param nr_procs: The number of processes that are to be used in total, including
            the manager.
        """
        super().__init__()
        self.queue = Queue()
        self.processes = [WorkerProcess(self.queue) for i in range(nr_procs - 1)]
        self.worker_manager = worker_manager

    def run(self, filename: str) -> None:
        """
        Processes a CSV file of documents using a number of worker processes.

        :param filename: The name of the CSV file; relative to the directory from where
            the code is being run.
        """
        # Start the worker processes
        for proc in self.processes:
            proc.process.start()
        nr_procs = len(self.processes) + 1

        for index, data in enumerate(read_csv(filename)):
            data = self.get_data(data)

            # If the manager isn't supposed to process documents, simply fill the queue
            # Else, make sure the workers can stay busy by keeping tasks in the queue
            # We use three times the number of processes as the buffer size
            if not self.worker_manager or self.queue.qsize() < 3 * nr_procs:
                self.queue.put(data)
            # If the queue contains enough items, then the manager can work on it
            else:
                self.process_data(*data)

            # TODO remove after testing
            if index == 9999:
                break

        # Signal to the workers that all of the documents have been distributed
        for _ in range(nr_procs - 1):
            self.queue.put(None)

        # TODO gather data

        # Wait for all of the workers to finish
        for proc in self.processes:
            proc.process.join()


def parse_arguments() -> Namespace:
    """
    Parses the command line arguments using the `argparse` library.

    :return: A namespace containing the parsed command line arguments.
    """
    parser = ArgumentParser()
    parser.add_argument("filename", nargs="?", default="data/news_dataset.csv")
    parser.add_argument("--threads", "-j", default=1, type=int)
    parser.add_argument("--worker-manager", "-w", action="store_true")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()

    # Force the manager to work if it's the only process
    if args.threads == 1:
        args.worker_manager = True

    start = time()
    manager = ManagerProcess(args.threads, args.worker_manager)
    manager.run(args.filename)
    end = time()

    t = end - start
    print(t, "seconds")
    print(int(t) // 60, "minutes", t % 60, "seconds")

    # 1:28:47.794076442718506 (single thread, full dataset, preprocessing + tf.idf)
