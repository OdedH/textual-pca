from torchvision import datasets
import random

class ImagesOnlyStanfordCars(datasets.StanfordCars):
    """
    A StanfordCars based dataset that return only images
    """
    def __getitem__(self, index: int):
        image, target = super().__getitem__(index)
        return image