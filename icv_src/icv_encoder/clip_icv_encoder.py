from .base_icv_encoder import BaseICVEncoder, ICVEncoderOutput
import torch
from loguru import logger
from torch import nn
from transformers import AutoConfig, AutoModel, CLIPProcessor, CLIPModel


class CLIPICVEncoder(BaseICVEncoder):
    def __init__(
        self,
        llm_hidden_dim,
        llm_layers,
        alpha_learnable=True,
        alpha_init_value=0.0,
        use_sigmoid=False,
        embedding_name=None,
        clip_path=None,
    ) -> None:
        super().__init__()
        self.clip_embed = CLIPModel.from_pretrained(clip_path)
        for param in self.clip_embed.parameters():
            param.requires_grad = False
        self.processor = CLIPProcessor.from_pretrained(clip_path)
        self.alpha = torch.nn.Parameter(
            torch.full(size=(1, llm_layers), fill_value=float(alpha_init_value)),
            requires_grad=alpha_learnable,
        )
        self.icv = nn.ModuleList(
            [torch.nn.Linear(512, llm_hidden_dim) for _ in range(llm_layers)]
        )

        # 均值为0，标准差为0.01的正态分布
        # self.icv = torch.nn.Linear(512,4096)

        # torch.nn.init.normal_(self.icv, mean=0.0, std=0.01)

        self.use_sigmoid = use_sigmoid

    def forward(self, data) -> ICVEncoderOutput:
        # len(data):batchsize
        batchsize = len(data)
        all_icv = []
        for i in range(batchsize):
            outs = self.clip_embed(
                pixel_values=data[i]["pixel_values"], input_ids=data[i]["input_ids"]
            )
            icv = [
                icv_layer(
                    torch.mean(
                        outs.text_model_output.last_hidden_state[
                            :,
                            -1,
                        ],
                        dim=0,
                        keepdim=True,
                    )
                )
                for icv_layer in self.icv
            ]
            all_icv.append(icv)
        final_icv = torch.stack(
            [torch.cat(sublist, dim=0) for sublist in all_icv], dim=0
        )
        return ICVEncoderOutput(
            in_context_vector=final_icv, alpha=self.get_alpha(), in_context_feature=None
        )

    def get_alpha(self):
        if self.use_sigmoid:
            return torch.sigmoid(self.alpha)
        return self.alpha
