import torch.cuda
import argparse
import os
os.chdir('./customized_BLIP')
from customized_BLIP.models.blip import image_set_blip_decoder
from customized_BLIP.models.blip_itm import normalized_blip_itm
from customized_BLIP.semantic_pca_utils import blip_load_image, blip_transform, calc_images_mean_embedding, \
    project_images_for_matching
from local_datasets.lsun_datasets import ImagesOnlyLSUN
from local_datasets.general_datasets import DatasetFromPath

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--image_size", type=int, default=384)  # Resize for BLIP
    parser.add_argument("--top_p", type=float, default=0.01)  # BLIP nucleus sampling
    parser.add_argument("--max_length", type=int, default=6)
    parser.add_argument("--min_length", type=int, default=1)
    parser.add_argument("--number_words_wordnet_similarity", type=int, default=12)
    parser.add_argument("--repetition_penalty", type=float, default=5.0)
    parser.add_argument("--variance_coeff", type=float, default=10.0)
    parser.add_argument("--logit_coeff", type=float, default=8.0)
    parser.add_argument("--number_principal_phrases", type=int, default=7)
    parser.add_argument("--prev_phrases_dist_coeff", type=float, default=-80.0)
    parser.add_argument("--k_for_topk_logits", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--number_of_batches", type=int, default=5)
    parser.add_argument("--dataset_path", type=str, default="./dataset/custom_dataset")

    args = parser.parse_args()

    return args


def generate_phrases(dataset, device, image_size, top_p,
                     max_length, min_length, repetition_penalty, variance_coeff, logit_coeff,
                     number_principal_phrases, prev_phrases_dist_coeff, k_for_topk_logits,
                     batch_size, number_of_batches, number_words_wordnet_similarity, **kwargs
                     ):
    """Function to generate the principal and average phrases"""

    with torch.no_grad():
        # create match model:
        model_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_large_retrieval_coco.pth'
        match_model = normalized_blip_itm(pretrained=model_url, image_size=image_size, vit='large')
        match_model.eval()
        match_model = match_model.to(device)

        # captioning model
        model_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_large_caption.pth'
        blip_cap_principal = image_set_blip_decoder(pretrained=model_url,
                                                    image_size=image_size,
                                                    vit='large',
                                                    k_for_topk_logits=k_for_topk_logits,
                                                    variance_coeff=variance_coeff,
                                                    prev_phrases_dist_coeff=prev_phrases_dist_coeff,
                                                    logit_coeff=logit_coeff,
                                                    number_words_wordnet_similarity=number_words_wordnet_similarity,
                                                    )
        blip_cap_principal.eval()
        blip_cap_principal = blip_cap_principal.to(device)
        image_embedding_mean = calc_images_mean_embedding(dataset=dataset,
                                                          batch_size=batch_size, number_of_batches=number_of_batches,
                                                          projector=blip_cap_principal.visual_encoder, device=device)

        mean_phrase = \
            blip_cap_principal.generate_mean_phrase(image_embedding_mean.to(device), sample=True, top_p=top_p,
                                                    max_length=max_length,
                                                    min_length=min_length, repetition_penalty=repetition_penalty,
                                                    )[0]
        mean_phrase = blip_cap_principal.postprocess_caption(mean_phrase, is_mean=True)

        # principal phrases
        images_encoding_matching = project_images_for_matching(dataset=dataset,
                                                               batch_size=batch_size,
                                                               number_of_batches=number_of_batches,
                                                               match_model=match_model, device=device)
        principal_phrases = blip_cap_principal.generate_principal_phrases(image_embedding_mean=image_embedding_mean,
                                                                          images_encoding_matching=images_encoding_matching,
                                                                          top_p=top_p,
                                                                          max_length=max_length,
                                                                          min_length=min_length,
                                                                          repetition_penalty=repetition_penalty,
                                                                          number_principal_phrases=number_principal_phrases,
                                                                          match_model=match_model,
                                                                          mean_phrase=mean_phrase,
                                                                          sample=True
                                                                          )
        principal_phrases = blip_cap_principal.principal_phrase_post_process(
            [blip_cap_principal.postprocess_caption(x) for x in principal_phrases])

        return mean_phrase, principal_phrases


def run(dataset, device):
    args = get_args()
    mean_phrase, phrases = generate_phrases(dataset=dataset, device=device, **vars(args))
    return mean_phrase, phrases


if __name__ == "__main__":
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    transform = blip_transform(blip_load_image, image_size=args.image_size, device=device)

    # Load many types of datasets or create your own using local_datasets
    # LSUN example
    dataset = ImagesOnlyLSUN(root='../data/lsun', classes=['church_outdoor_train'], transform=transform)

    # To use a custom dataset:
    # dataset = DatasetFromPath(root=args.dataset_path, transform=transform)

    mean_phrase, principal_phrases = run(dataset, device)
    print(f"avg phrase: {mean_phrase}")
    print(f"principal phrases: {principal_phrases}")

    print()
