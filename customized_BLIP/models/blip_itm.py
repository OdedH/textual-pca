import sys

sys.path.insert(0, '..')
from customized_BLIP.models.med import BertConfig, BertModel
from typing import List
import torch
from torch import nn
import torch.nn.functional as F
from customized_BLIP.models.blip import create_vit, init_tokenizer, load_checkpoint


class Normalized_BLIP_ITM(nn.Module):
    """
    BLIP ITM is a matching head that is used to calculate the similarity between images and texts.
    We modify BLIP's matching head such that, similarly to PCA, it will center the data with the mean sentence.
    """

    def __init__(self,
                 med_config='configs/med_config.json',
                 image_size=384,
                 vit='base',
                 vit_grad_ckpt=False,
                 vit_ckpt_layer=0,
                 embed_dim=256,
                 device="cuda"
                 ):
        """
        Args:
            med_config (str): path for the mixture of encoder-decoder model's configuration file
            image_size (int): input image size
            vit (str): model size of vision transformer
        """
        super().__init__()

        self.visual_encoder, vision_width = create_vit(vit, image_size, vit_grad_ckpt, vit_ckpt_layer)
        self.tokenizer = init_tokenizer()
        med_config = BertConfig.from_json_file(med_config)
        med_config.encoder_width = vision_width
        self.text_encoder = BertModel(config=med_config, add_pooling_layer=False)

        text_width = self.text_encoder.config.hidden_size

        self.vision_proj = nn.Linear(vision_width, embed_dim)
        self.text_proj = nn.Linear(text_width, embed_dim)

        self.itm_head = nn.Linear(text_width, 2)
        self.device = device

    def forward(self, image_embeds: torch.Tensor, caption: List[str], mean_sentence: str) -> torch.Tensor:
        """
        A forward function of a BLIP_M - A modified version of BLIP's ITM matching head
        1. Embedding images and texts to a shared visual-language space
        2. Similar to PCA, normalize by subtracting the embedding of the average sentence from the texts embeddings
        3. Calculating cosine similarity

        :param image_embeds: Image embedding in the shape of (number of images, embedding size) of images to calculate
                             matching scores with
        :param caption: texts to calculate matching scores with
        :param mean_sentence: the mean (average) sentence
        :return: A similarity matrix of the shape (number of images, number of texts)
        """
        text = self.tokenizer(caption, padding='max_length', truncation=True, max_length=35,
                              return_tensors="pt").to(image_embeds.device)
        avg = self.tokenizer(mean_sentence, padding='max_length', truncation=True, max_length=35,
                             return_tensors="pt").to(image_embeds.device)

        text_output = self.text_encoder(text.input_ids, attention_mask=text.attention_mask,
                                        return_dict=True, mode='text')
        avg_text_output = self.text_encoder(avg.input_ids, attention_mask=avg.attention_mask,
                                            return_dict=True, mode='text')

        text_feat = F.normalize(self.text_proj(text_output.last_hidden_state[:, 0, :]), dim=-1)
        avg_text_feat = F.normalize(self.text_proj(avg_text_output.last_hidden_state[:, 0, :]), dim=-1)

        normed_text = text_feat - avg_text_feat

        normed_text = F.normalize(normed_text, dim=-1)
        image_feat = F.normalize(image_embeds, dim=-1)
        sim = image_feat @ normed_text.t()
        return sim

    def get_texts_projection(self, texts: List[str], max_length=35) -> List[torch.Tensor]:
        """
        For embedding the text in BLIP's vision-language space we need to:
        1. Encode the text with BLIP
        2. Project the embedding into BLIP's vision-language space
        :param texts: list of texts tp project
        :param max_length: exact default value is not significant and is larger than actual length
        """
        with torch.no_grad():
            text = self.tokenizer(texts, padding='max_length', truncation=True, max_length=max_length,
                                  return_tensors="pt").to(self.device)
            text_output = self.text_encoder(text.input_ids, attention_mask=text.attention_mask,
                                            return_dict=True, mode='text')
            text_feat = F.normalize(self.text_proj(text_output.last_hidden_state[:, 0, :]), dim=-1)
        return text_feat

    def get_images_projection(self, images):
        with torch.no_grad():
            image_embeds = self.visual_encoder(images)
            projected_data = self.vision_proj(image_embeds[:, 0, :])
        return F.normalize(projected_data, dim=-1)


def normalized_blip_itm(pretrained='', **kwargs):
    """A c'tor like function similar to blip_itm function in original BLIP"""
    model = Normalized_BLIP_ITM(**kwargs)
    if pretrained:
        model, msg = load_checkpoint(model, pretrained)
    return model
