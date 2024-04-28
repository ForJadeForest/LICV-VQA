from .base_icv_encoder import BaseICVEncoder, ICVEncoderOutput
import torch
from loguru import logger
from torch import nn
from transformers import (
    AutoConfig,
    AutoModel,
)


class GlobalICVEncoder(BaseICVEncoder):
    def __init__(
        self,
        llm_hidden_dim,
        llm_layers,
        alpha_learnable=True,
        alpha_init_value=0.0,
        use_sigmoid=False,
    ) -> None:
        super().__init__()

        self.alpha = torch.nn.Parameter(
            torch.full(size=(1, llm_layers), fill_value=float(alpha_init_value)),
            requires_grad=alpha_learnable,
        )
        self.icv = torch.nn.Parameter(torch.empty(1, llm_layers, llm_hidden_dim))
        # 均值为0，标准差为0.01的正态分布
        torch.nn.init.normal_(self.icv, mean=0.0, std=0.01)

        self.use_sigmoid = use_sigmoid

    def forward(self) -> ICVEncoderOutput:
        return ICVEncoderOutput(
            in_context_vector=self.icv, alpha=self.get_alpha(), in_context_feature=None
        )

    def get_alpha(self):
        if self.use_sigmoid:
            return torch.sigmoid(self.alpha)
        return self.alpha
