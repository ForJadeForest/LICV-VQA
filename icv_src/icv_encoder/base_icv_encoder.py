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
