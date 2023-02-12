import torch.cuda
import sys
sys.path.insert(0, '..')
from local_datasets.lsun_datasets import ImagesOnlyLSUN
import matplotlib.pyplot as plt
import random
from BLIP.semantic_pca_utils import blip_transform, blip_load_image, draw_images_dataset,UnNormalize


if __name__ == "__main__":
    random.seed(42)
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    transform = blip_transform(blip_load_image, image_size=384, device="cuda:0")
    # celeba_image_only_raw = ImagesOnlyCelebA(root="../data/celebA", split='train', transform=transform)
    # lsun_dataset = ImagesOnlyLSUN(root='../data/lsun', classes=['kitchen_train'], transform=transform)
    # coco_cluster80_5 = DatasetFromPath(root="../data/coco_hierarchical_clustering/80/4/", transform=transform)
    # gan_house_dalle = DatasetFromPath(root="../data/gans/house/DALLE-mega/", transform=transform)

    # current_dataset = ImagesOnlyLSUN(root='../data/lsun', classes=['church_outdoor_train'], transform=transform)
    # current_dataset = DatasetFromPath(root="../data/coco_hierarchical_clustering/80/4/", transform=transform)
    # current_dataset = ImagesOnlyLSUN(root='../data/lsun', classes=['kitchen_train'], transform=transform)
    current_dataset = ImagesOnlyLSUN(root='../data/lsun', classes=['bridge_train'], transform=transform)
    # current_dataset = DatasetFromPath(root="../data/gans/house/DALLE-mega/", transform=transform)
    # current_dataset = DatasetFromPath(root="../data/coco_hierarchical_clustering/80/5/", transform=transform)
    # cars_dataset = ImagesOnlyStanfordCars(root='../data/stanfordCars', split='train', transform=transform)
    unnorm = UnNormalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
    # embeds, images = get_multiple_images_embedding_and_images(celeba_image_only_raw,5, 100, model, device)
    draw_images_dataset(7, current_dataset, unnorm)
    name = "bridge"
    plt.savefig(rf'./figures/set_images/{name}.png')

    plt.show()