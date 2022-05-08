# Use LMDB library to create lightweight datasets.
# More info at: https://lmdb.readthedocs.io/en/release/

import os
from typing import Any, Dict, List

import lmdb


class LMDBGenerator:
    def __init__(self, buffer_size: int = 1000):
        """Class constructor.

        Keyword Arguments:
            buffer_size {int} -- Buffer size to save on disk (default: {1000}).
        """

        self.buffer_size = buffer_size

    def _write_buffer(self, buffer: Dict[Any], lmdb_dataset: object, free_buffer: bool = False):
        """Write a buffer in a LMDB dataset (i.e. LMDB file).

        Arguments:
            buffer {Dict[any]} -- A buffer contaning data and their respective annotations.
            lmdb_dataset {object} -- The LMDB dataset file.

        Keyword Arguments:
            free_buffer {bool} -- Whether free the buffer dictionary (default: {False}).
        """

        with lmdb_dataset.begin(write=True) as dataset:
            for key, value in buffer.items():
                dataset.put(key, value)

        if free_buffer:
            buffer = {}

    def create_dataset(self, data: List[Any], annotations: List[Any], output_path: str):
        """Create a LMDB dataset.

        Arguments:
            data {List[Any]} -- List of data to be kept in the dataset.
            annotations {List[Any]} -- List of annotations to be kept in the dataset.
            output_path {str} -- Path to save the LMDB file.
        """

        assert len(data) == len(
            annotations
        ), "Number of samples does not match with the number of annotations."

        buffer = {}
        lmdb_dataset = lmdb.open(output_path)

        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Encode samples
        for i in range(len(data)):
            buffer["image-%09d" % i] = data[i]
            buffer["label-%09d" % i] = annotations[i]

            if i % self.buffer_size == 0 or i == len(data) - 1:
                self._write_buffer(buffer, lmdb_dataset, free_buffer=True)
