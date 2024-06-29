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
        clip_ft,
        alpha_learnable=True,
        alpha_init_value=0.0,
        use_sigmoid=False,
        embedding_name=None,
        clip_path=None,
    ) -> None:
        super().__init__()
        self.clip_embed = CLIPModel.from_pretrained(clip_path)
        for param in self.clip_embed.parameters():
            if clip_ft==True:
              param.requires_grad = True
            else:
              param.requires_grad = False
        self.processor = CLIPProcessor.from_pretrained(clip_path)
        self.alpha = torch.nn.Parameter(
            torch.full(size=(1, llm_layers), fill_value=float(alpha_init_value)),
            requires_grad=alpha_learnable,
        )
        self.icv = nn.ModuleList(
            [torch.nn.Linear(512, llm_hidden_dim) for _ in range(llm_layers)]
        )

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
    
    def load_weight(self,path,load_clip):
        icv_weights = torch.load(path)
        for i, layer in enumerate(self.icv):
            layer_state_dict = {
                        'weight': icv_weights[f'icv_encoder.icv.{i}.weight'],
                        'bias': icv_weights[f'icv_encoder.icv.{i}.bias']
                    }
            layer.load_state_dict(layer_state_dict)
        self.alpha = torch.nn.Parameter(icv_weights['icv_encoder.alpha'], requires_grad=True)
        if load_clip==True:
            new_state_dict = {}
            for key, value in icv_weights.items():
                new_key = key.replace('icv_encoder.clip_embed.', '')
                if new_key in icv_weights:
                    new_state_dict[new_key] = value
            self.clip_embed.load_state_dict(new_state_dict, strict=False)
        else:
            pass