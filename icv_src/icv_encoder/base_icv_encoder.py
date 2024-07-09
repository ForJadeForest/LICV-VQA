import torch
from loguru import logger
from torch import nn
from transformers import AutoConfig, AutoModel, AutoTokenizer
from dataclasses import dataclass
from typing import Optional


@dataclass
class ICVEncoderOutput:
    in_context_feature: Optional[torch.FloatTensor]
    in_context_vector: Optional[torch.FloatTensor]
    alpha: Optional[torch.FloatTensor]


class BaseICVEncoder(nn.Module):
    def __init__(
        self,
    ) -> None:
        super().__init__()
        self.alpha = None
        self.icv_encoder = None

    def forward(self, *args, **kwargs) -> ICVEncoderOutput:
        raise NotImplemented
    
def torch_PCA(data, k):
    data = data.T
    mean = torch.mean(data, axis=0)
    data_centered = data - mean
    cov_matrix = torch.matmul(data_centered.T, data_centered) / (data.shape[0] - 1)
    U,S,V = torch.svd(cov_matrix.float())
    principal_components = U[:, :k]
    data_transformed = torch.mm(data_centered.float(), principal_components)
    return data_transformed.T