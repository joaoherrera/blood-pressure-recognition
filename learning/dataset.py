import json
from datetime import date
from typing import Dict

import numpy as np


class AnnotationFile:
    def __init__(self, annotation_file_path: str) -> None:
        self.filepath = annotation_file_path
        self.new()

    def new(self) -> None:
        self.tree = {"info": {}, "licenses": [], "categories": [], "images": [], "annotations": []}

    def save(self) -> None:
        with open(self.filepath, "w") as annotation_file:
            json.dump(self.tree, annotation_file)

    def load(self) -> None:
        with open(self.filepath, "r") as annotation_file:
            self.tree = json.load(annotation_file)

    def insert_info(self) -> None:
        info_data = {
            "year": date.today().year(),
            "version": "0.1",
            "description": "",
            "date_created": date.today(),
        }

        self.tree["info"].append(info_data)

    def insert_image(self, image_name: str, image: np.ndarray) -> None:
        image_data = {
            "id": len(self.tree["images"]) + 1,
            "file_name": image_name,
            "width": image.shape[0],
            "height": image.shape[1],
        }

        self.tree["images"].append(image_data)

    def insert_annotation(self, image_id: int, category_id: int, data: Dict) -> None:
        annotation_data = {
            "id": len(self.tree["annotations"]) + 1,
            "image_id": image_id,
            "category_id": category_id,
        }

        self.tree["annotations"].append({**annotation_data, **data})

    def insert_category(self, category_name: str) -> None:
        category_data = {
            "id": len(self.tree["categories"]) + 1,
            "name": category_name,
            "supercategory": "",
        }

        self.tree["categories"].append(category_data)


class AnnotationHandler:
    pass
