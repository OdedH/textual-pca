from torchvision import datasets
import random
from typing import Any, Callable, Optional, Tuple, List


class ImagesOnlyCOCO(datasets.CocoCaptions):
    """
    A COCO based dataset that return only images
    """

    def __getitem__(self, index: int):
        image, target = super().__getitem__(index)
        return image


class ShortImagesOnlyCOCO(ImagesOnlyCOCO):
    """
    A short (trimmed) ImagesOnlyCOCO dataset for experimenting
    """

    def __init__(self, *args, **kwargs):
        """
        Size will be the size of the dataset
        :param args:
        :param kwargs:
        """
        self.size = kwargs.pop('size')
        super(ImagesOnlyCOCO, self).__init__(*args, **kwargs)

    def __len__(self):
        return self.size

class OneAnswerCOCO(datasets.CocoCaptions):
    """`MS Coco Captions <https://cocodataset.org/#captions-2015>`_ Dataset.
    A COCO dataset with only one random answer selected
    Args:
        root (string): Root directory where images are downloaded to.
        annFile (string): Path to json annotation file.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.ToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        transforms (callable, optional): A function/transform that takes input sample and its target as entry
            and returns a transformed version.
    """
    def _load_target(self, id: int) -> str:
        return random.choice(super()._load_target(id))

class ShortOneAnswerCOCO(OneAnswerCOCO):
    """`MS Coco Captions <https://cocodataset.org/#captions-2015>`_ Dataset.
    A COCO dataset with only one random answer selected
    Args:
        root (string): Root directory where images are downloaded to.
        annFile (string): Path to json annotation file.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.ToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        transforms (callable, optional): A function/transform that takes input sample and its target as entry
            and returns a transformed version.
    """
    def __init__(self, *args, **kwargs):
        """
        Size will be the size of the dataset
        :param args:
        :param kwargs:
        """
        self.size = kwargs.pop('size')
        super(OneAnswerCOCO, self).__init__(*args, **kwargs)

    def __len__(self):
        return self.size
    