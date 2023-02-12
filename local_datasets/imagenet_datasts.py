from torchvision import datasets
import random

class ImagesOnlyImagenetClass(datasets.ImageNet):
    """
    Image Only ImageNet dataset of size num_samples that contains only samples from 'target' class
    """
    def __init__(self, *args, **kwargs):
        self.selected_target = kwargs.pop('target')
        self.num_samples =  kwargs.pop('num_samples')
        super().__init__(*args, **kwargs)
        self.idx_list = []
        # idx = list(range(super().__len__()))
        idx = list(range(1300*self.selected_target,1300*(self.selected_target+1)))
        random.shuffle(idx)
        for i in idx:
            if super().__getitem__(i)[1] == self.selected_target:
                self.idx_list.append(i)
            if len(self.idx_list) >= self.num_samples:
                break
        assert len(self.idx_list) == self.num_samples, "Not enough samples from requested class"

    def __getitem__(self, index: int):
        image, target = super().__getitem__(self.idx_list[index])
        return image

    def __len__(self):
        return len(self.idx_list)