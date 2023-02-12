import torch
from torch.utils.data import DataLoader, Dataset
from matplotlib import pyplot as plt
from nltk.corpus import wordnet as wn
import torchvision.transforms as transforms
import random
from torchvision.transforms.functional import InterpolationMode


def calc_images_mean_embedding(dataset, batch_size, number_of_batches, projector, device):
    """
    A function to generate mean embedding later to be used to generate mean sentence and a centered matching head
    denoted in the article as BLIP_M.
    """
    acc = []
    for data in DataLoader(dataset, batch_size=batch_size):
        if len(data.shape) == 5:  # this means that every batch is a collection of images
            assert batch_size == 1, "if batch is a collection of images batch size of one is expected"
            data = data[0]
        with torch.no_grad():
            projected_data = projector(data.to(device)).detach()
            acc.append(projected_data.to("cpu"))
            torch.cuda.empty_cache()
        number_of_batches -= 1
        if number_of_batches == 0:
            break
    return torch.mean(torch.cat(acc), dim=0).unsqueeze(0).to(device)


def project_images_for_matching(dataset, batch_size, number_of_batches, match_model, device):
    """ This function creates an images embedding mean later to be used by the matching head"""
    acc = []
    for data in DataLoader(dataset, batch_size=batch_size):
        if len(data.shape) == 5:  # this means that every batch is a collection of images
            assert batch_size == 1, "if batch is a collection of images batch size of one is expected"
            data = data[0]
        with torch.no_grad():
            image_embeds = match_model.visual_encoder(data.to(device)).detach()
            projected_data = match_model.vision_proj(image_embeds[:, 0, :])
            acc.append(projected_data.to("cpu"))
            torch.cuda.empty_cache()
        number_of_batches -= 1
        if number_of_batches == 0:
            break
    return torch.cat(acc).to(device)


def principal_sentence_post_process(sentences):
    sentences_post = []
    for sentence in sentences:
        words = []
        for word in sentence.split():
            if word in words: continue
            words.append(word)
        sentence = " ".join(words)
        if sentence in sentences_post:
            continue
        if sentence == "":
            continue
        sentences_post.append(sentence)
    return sentences_post


def postprocess_caption(caption):
    caption = remove_words_without_content(caption)
    return caption


def remove_words_without_content(caption):
    words = []
    for word in caption.split():
        word = (''.join([ch for ch in word if ch.isalnum()]))
        if wn.synsets(word) and len(word) > 2:
            words.append(word)
    return " ".join(words)


class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor


def blip_transform(blip_load_image_function, image_size, device):
    def func(image):
        return (blip_load_image_function(image, image_size, device))

    return func


def blip_load_image(image, image_size, device):
    raw_image = image

    w, h = raw_image.size

    transform = transforms.Compose([
        transforms.Resize((image_size, image_size), interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    ])
    image = transform(raw_image)
    return image


def draw_images_dataset(grid_size, dataset, unnorm):
    f, axarr = plt.subplots(grid_size, grid_size, figsize=(16, 16))
    random_indices = random.sample(range(len(dataset)), grid_size ** 2 + 1)
    for i in range(grid_size):
        for j in range(grid_size):
            image = unnorm(dataset[random_indices[i * grid_size + j]]).squeeze().permute(1, 2, 0)
            axarr[i, j].imshow(image)
            axarr[i, j].axis('off')
    f.subplots_adjust(wspace=0, hspace=0)
    f.patch.set_visible(False)
    plt.axis('off')
    f.tight_layout()


def chunker(seq, size):
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))


def calc_word_similarity(model, batch_size, k_similar_words):
    word_list = list(model.tokenizer.vocab.keys())
    embeddings = []
    for words in chunker(word_list, batch_size):
        embeddings.append(model.get_texts_projection(words).detach())
    embeddings = torch.vstack(embeddings)

    similarities = []
    for token in range(embeddings.size(0)):
        similarities.append(torch.topk(embeddings[token] @ embeddings.t(), k=k_similar_words))
    return similarities


def remove_mean_sentence_words(caption, mean):
    return " ".join([word for word in caption.split() if word not in mean])


def show_image(img):
    plt.imshow(img.to("cpu").squeeze().permute(1, 2, 0))
    plt.axis('off')
    plt.show()

