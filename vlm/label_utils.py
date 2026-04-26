from typing import Dict


MP3D_LABEL_MAP: Dict[str, str] = {
    "chair": "chair",
    "table": "table | dining table | coffee table | desk",
    "picture": "picture | framed photograph",
    "cabinet": "cabinet",
    "cushion": "pillow",
    "sofa": "couch",
    "bed": "bed",
    "chest_of_drawers": "nightstand",
    "plant": "potted plant",
    "sink": "sink",
    "toilet": "toilet",
    "stool": "stool",
    "towel": "towel",
    "tv_monitor": "tv",
    "shower": "shower",
    "bathtub": "bathtub",
    "counter": "counter",
    "fireplace": "fireplace",
    "gym_equipment": "gym equipment",
    "seating": "seating",
    "clothes": "clothes",
}

HM3D_LABEL_MAP: Dict[str, str] = {
    "chair": "chair",
    "bed": "bed",
    "plant": "potted plant",
    "toilet": "toilet",
    "tv_monitor": "tv",
    "sofa": "couch",
}


def normalize_objectnav_label(label: str, dataset_path: str) -> str:
    dataset_path = dataset_path.lower()

    if "/mp3d/" in dataset_path:
        return MP3D_LABEL_MAP.get(label, label)

    if "/hm3d/" in dataset_path:
        return HM3D_LABEL_MAP.get(label, label)

    return label
