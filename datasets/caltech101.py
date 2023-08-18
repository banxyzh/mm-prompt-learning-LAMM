import os
import pickle

from dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase
from dassl.utils import mkdir_if_missing

from .oxford_pets import OxfordPets
from .dtd import DescribableTextures as DTD

IGNORED = ["BACKGROUND_Google", "Faces_easy"]
NEW_CNAMES = {
    "airplanes": "airplane",
    "Faces": "face",
    "Leopards": "leopard",
    "Motorbikes": "motorbike",
}


@DATASET_REGISTRY.register()
class Caltech101(DatasetBase):

    dataset_dir = "caltech-101"

    def __init__(self, cfg):
        root = os.path.abspath(os.path.expanduser(cfg.DATASET.RO