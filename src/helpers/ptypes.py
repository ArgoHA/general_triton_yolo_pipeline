from typing import Generic, List, Literal, Protocol, Tuple, TypedDict, TypeVar, get_args
import yaml
import numpy as np


class YOLOScores(TypedDict):
    boxes: np.ndarray[np.ndarray]
    scores: np.ndarray[float]
    class_ids: np.ndarray[int]


color_mapping = {1: (245, 135, 66), 2: (0, 255, 0), 3: (37, 190, 73)}


with open("config.yaml", "r") as cfg:
    try:
        cfg = yaml.safe_load(cfg)
    except yaml.YAMLError as exc:
        print(exc)

name_to_label_mapping = cfg["name_to_label_mapping"]
label_to_name_mapping = {v: k for k, v in name_to_label_mapping.items()}
