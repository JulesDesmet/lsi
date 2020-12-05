#!/usr/bin/env python

from multiprocessing import Process, Queue, set_start_method

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

    # The worker processes this manager manages
    processes: list[WorkerProcess]
    # The queues through which documents are sent to the workers, associated by index
    queues: list[Queue]

    def __init__(self, nr_procs: int = 1):
        """
        Initialises the worker processes and their queues.

        :param nr_procs: The number of processes that are to be used in total, including
            the manager.
        """
        super().__init__()
        self.queues = [Queue() for _ in range(nr_procs - 1)]
        self.processes = [WorkerProcess(self.queues[i]) for i in range(nr_procs - 1)]

    def run(self, filename: str) -> None:
        """
        Processes a CSV file of documents using a number of worker processes.

        :param filename: The name of the CSV file; relative to the directory from where
            the code is being run.
        """
        # Start the worker processes
        for proc in self.processes:
            proc.process.start()

        for index, data in enumerate(read_csv(filename)):
            proc_id = index % (len(self.processes) + 1)
            data_id, content = self.get_data(data)

            # Process one out of every `len(self.processes) + 1` in this process
            if proc_id == len(self.processes):
                self.process_data(data_id, content)
            # Distribute the other `len(self.processes)` evenly to the workers
            else:
                self.queues[proc_id].put((data_id, content))

            # TODO remove after testing
            if index == 9999:
                break

        # Signal to the workers that all of the documents have been distributed
        for queue in self.queues:
            queue.put(None)

        # TODO gather data

        # Wait for all of the workers to finish
        for proc in self.processes:
            proc.process.join()


if __name__ == "__main__":

    # Allow some arguments for this script (in this exact order without leaving any out)
    # 1. A dataset in CSV format            (default = data/news_dataset.csv)
    from sys import argv

    filename = argv[1] if len(argv) >= 2 else "data/news_dataset.csv"

    from time import time

    start = time()
    manager = ManagerProcess(12)
    manager.run(filename)
    end = time()

    t = end - start
    print(t, "seconds")
    print(int(t) // 60, "minutes", t % 60, "seconds")
    # 1:28:47.794076442718506 (single threaded preprocessing + tf.idf)
