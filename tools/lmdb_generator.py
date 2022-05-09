# Use LMDB library to create lightweight datasets.
# More info at: https://lmdb.readthedocs.io/en/release/

import os
from argparse import ArgumentParser
from glob import glob
from itertools import chain
from typing import Any, Dict, List

import lmdb


class LMDBGenerator:
    def __init__(self, buffer_size: int = 1000):
        """Class constructor.

        Keyword Arguments:
            buffer_size {int} -- Buffer size to save on disk (default: {1000}).
        """

        self.buffer_size = buffer_size

    def _write_buffer(self, buffer: Dict, lmdb_dataset: object, free_buffer: bool = False):
        """Write a buffer in a LMDB dataset (i.e. LMDB file).

        Arguments:
            buffer {Dict[any]} -- A buffer contaning data and their respective annotations.
            lmdb_dataset {object} -- The LMDB dataset file.

        Keyword Arguments:
            free_buffer {bool} -- Whether free the buffer dictionary (default: {False}).
        """

        with lmdb_dataset.begin(write=True) as dataset:
            for key, value in buffer.items():
                dataset.put(str(key).encode(), str(value).encode())

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
        lmdb_dataset = lmdb.open(output_path, map_size=int(1e9))

        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Encode samples
        for i in range(len(data)):
            buffer["image-%09d" % i] = data[i]
            buffer["label-%09d" % i] = annotations[i]

            if i % self.buffer_size == 0 or i == len(data) - 1:
                self._write_buffer(buffer, lmdb_dataset, free_buffer=True)


class LMDBImageGenerator(LMDBGenerator):
    @staticmethod
    def read_images_from_directory(
        directory_path: str, extensions: List[str] = ["jpg", "png"]
    ) -> List[object]:
        """Read images from a given directory and return binary files instead of arrays of pixels.

        Arguments:
            directory_path {str} -- Path to a directory containing image files.

        Keyword Arguments:
            extensions {List[str]} -- File extensions to be considered (default: {["jpg", "png"]}).

        Raises:
            NotADirectoryError: When the `directory_path` does not exist.

        Returns:
            List[object] -- A list of images as binary files.
        """

        if not os.path.exists(directory_path):
            raise NotADirectoryError()

        images = []

        # Read image paths from directory
        image_paths = chain([glob(os.path.join(directory_path, f"*.{ext}")) for ext in extensions])
        image_paths = list(image_paths)[0]

        # Open files as binary
        sorted(image_paths)

        for image_path in image_paths:
            with open(image_path, "rb") as image_file:
                binary_image = image_file.read()
                images.append(binary_image)

        return images

    @staticmethod
    def read_annotations_from_filename(
        directory_path: str, extensions: List[str] = ["jpg", "png"]
    ) -> List[str]:
        """Read annotations from image files. The annotations should be in the file name.
        In this method file names with the following pattern are considered:

        <annotation>_<filename>.<extension>
        Example: 9598098491_01.jpg, 9303974519_02.jpg, 2404269934_03.jpg, etc.


        Arguments:
            directory_path {str} -- Path to a directory containing image files.

        Keyword Arguments:
            extensions {List[str]} -- File extensions to be considered  (default: {["jpg", "png"]})

        Raises:
            NotADirectoryError: When the `directory_path` does not exist.

        Returns:
            List[str] -- A list of annotations.
        """

        if not os.path.exists(directory_path):
            raise NotADirectoryError()

        annotations = []

        # Read image paths from directory
        image_paths = chain([glob(os.path.join(directory_path, f"*.{ext}")) for ext in extensions])
        image_paths = list(image_paths)[0]

        # Extract values from image path.
        sorted(image_paths)

        for image_path in image_paths:
            image_name = os.path.splitext(os.path.basename(image_path))[0]

            # Here we assume that the image name is composed by <annotation>_<filename>.<extension>
            annotation = image_name.split("_")[0]
            annotations.append(annotation)

        return annotations


def build_arg_parser() -> ArgumentParser:
    """Create the arguments to generate a LMDB dataset."""
    parser = ArgumentParser()

    parser.add_argument(
        "--data-directory-path",
        type=str,
        help="Path to the files to be read.",
        required=True,
    )

    parser.add_argument(
        "--output-dataset-path",
        type=str,
        help="Path to the LMDB file.",
        required=True,
    )

    parser.add_argument(
        "--annotations-directory-path",
        type=str,
        help="""Path to the files with annotations labels. If not provided, 
        the --data-directory-path will be used.""",
    )

    parser.add_argument(
        "--extensions",
        type=str,
        nargs="+",
        help="File extensions to be considered. By default `jpg` and `png`.",
        default=["jpg", "png"],
    )

    return parser


if __name__ == "__main__":
    args = build_arg_parser().parse_args()
    lmdb_generator = LMDBImageGenerator()

    images = lmdb_generator.read_images_from_directory(args.data_directory_path)

    annotations = lmdb_generator.read_annotations_from_filename(
        args.data_directory_path
        if args.annotations_directory_path is None
        else args.annotations_directory_path
    )

    lmdb_generator.create_dataset(images, annotations, args.output_dataset_path)
