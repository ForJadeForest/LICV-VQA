import os
import os.path
from contextlib import suppress
from typing import List

from PIL import Image
from pycocotools.coco import COCO
from torchvision.datasets import VisionDataset


class CocoDataset(VisionDataset):
    def __init__(self, root: str, annFile: str) -> None:
        super().__init__(root)
        self.coco = COCO(annFile)
        self.ids = list(sorted(self.coco.imgs.keys()))

    def _load_image(self, id: int) -> Image:
        path = self.coco.loadImgs(id)[0]["file_name"]
        return os.path.join(self.root, path)

    def _load_target(self, id: int) -> List[str]:
        return [ann["caption"] for ann in self.coco.loadAnns(self.coco.getAnnIds(id))]

    def __getitem__(self, index: int):
        idx = self.ids[index]
        image = self._load_image(idx)
        target = self._load_target(idx)

        return {
            "single_caption": target[0],
            "image": image,
            "idx": index,
            "image_id": idx,
            "captions": target,
        }

    def __len__(self) -> int:
        return len(self.ids)
