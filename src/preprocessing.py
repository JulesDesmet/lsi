#!/usr/bin/env python

import csv
from typing import Generator
from sys import argv


def read_csv(filename: str) -> Generator[list[str], None, None]:
    """
    Reads a CSV file line by line. This method should be used as follows:
    ```
    for row in read_csv(...):
        do_something_with_row(...)
    ```

    :param filename: The CSV file, relative from the directory from where the script is
        being run, unless it's an absolute path.
    :return: A generator that yields the rows of the CSV file as a list of strings.
    """
    with open(filename) as csv_file:
        yield from csv.reader(csv_file)


if __name__ == "__main__":
    for index, line in enumerate(read_csv(argv[1])):
        print(line)
        if index == 10:
            break
