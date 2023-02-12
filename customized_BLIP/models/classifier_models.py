import torch.nn as nn
import torch

class BLIP_vec_classifier(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):

        super().__init__()
        self.input_sim = input_dim
        self.output_dim = output_dim

        self.multy_label_classifier = nn.Sequential(
            nn.Linear(input_dim,512),
            nn.BatchNorm1d(512),
            nn.Dropout(p=0.2),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.Dropout(p=0.2),
            nn.Linear(256,output_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.multy_label_classifier(x)