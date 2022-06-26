import os
from abc import ABC, abstractmethod
from glob import glob
from typing import Any, List

import cv2
import numpy as np
from torch import Module
from torch.utils.data import Dataset


class BaseTrainer(ABC):
    def __init__(self, device: str, model: Module) -> None:
        self.device = device
        self.model = model

        self.model.to(self.device)

    @abstractmethod
    def train():
        pass

    def validate():
        pass


class BaseDataset(ABC, Dataset):
    def __init__(self) -> None:
        super().__init__()

    def __getitem__(self, index: Any) -> None:
        return super().__getitem__(index)

    def __len__(self):
        pass

    @staticmethod
    def load_image(image_path: str, mode: int = cv2.IMREAD_COLOR) -> np.ndarray:
        """Read an image from disk.

        Args:
            image_path (str): Path to the image to be read.
            mode (int, optional): Read mode. Defaults to cv2.IMREAD_COLOR.

        Returns:
            np.ndarray: An array of pixels.
        """

        return cv2.imread(image_path, mode)

    @staticmethod
    def get_images_from_directory(directory_path: str, extensions: List[str] = ["jpg", "png"]) -> List[str]:
        """Read images from a given directory.

        Args:
            directory_path (str): Path to the directory where images are located.
            extensions (List[str], optional): Extensions. Defaults to ["jpg", "png"].

        Returns:
            List[str]: A list of image paths.
        """

        VALID_IMAGE_EXTENSIONS = ["jpg", "png", "tif", "jpeg", "tiff", "bmp"]

        assert [extension in VALID_IMAGE_EXTENSIONS for extension in extensions], f"Invalid extensions {extensions}"

        images_paths = []
        for extension in extensions:
            images_paths.extend(glob(os.path.join(directory_path, f"*.{extension}".upper())))
            images_paths.extend(glob(os.path.join(directory_path, f"*.{extension}".lower())))

        return images_paths
