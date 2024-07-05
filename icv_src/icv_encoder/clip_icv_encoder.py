from .base_icv_encoder import BaseICVEncoder, ICVEncoderOutput
import torch
from loguru import logger
from torch import nn
from transformers import AutoConfig, AutoModel, CLIPProcessor, CLIPModel
from sklearn.decomposition import PCA

class CLIPICVEncoder(BaseICVEncoder):
    def __init__(
        self,
        llm_hidden_dim,
        llm_layers,
        clip_ft,
        feature_mode,
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
        self.feature_mode = feature_mode
        if self.feature_mode == 4:
            self.icv = nn.ModuleList(
            [torch.nn.Linear(1024, llm_hidden_dim) for _ in range(llm_layers)]
        )
        else:
            self.icv = nn.ModuleList(
                [torch.nn.Linear(512, llm_hidden_dim) for _ in range(llm_layers)]
            )
        self.clip_ft = clip_ft
        self.use_sigmoid = use_sigmoid
        if self.feature_mode == 2:
            self.image_weight = torch.nn.Parameter(torch.full(size=(1, 32), fill_value=1/32, requires_grad=True))
            self.text_weight = torch.nn.Parameter(torch.full(size=(1, 33), fill_value=1/32, requires_grad=True))
        elif self.feature_mode == 3:
            self.pca = PCA(n_components=1)
        print("extractor mode:{}".format(self.feature_mode))

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
                    self.feature_extractor(outs.text_embeds, outs.image_embeds,self.feature_mode)
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
    
    def load_weight(self,path):
        icv_weights = torch.load(path)
        for i, layer in enumerate(self.icv):
            layer_state_dict = {
                        'weight': icv_weights[f'icv_encoder.icv.{i}.weight'],
                        'bias': icv_weights[f'icv_encoder.icv.{i}.bias']
                    }
            layer.load_state_dict(layer_state_dict)
        self.alpha = torch.nn.Parameter(icv_weights['icv_encoder.alpha'], requires_grad=True)
        if icv_weights["clip_ft"] == True:
            new_state_dict = {}
            for key, value in icv_weights.items():
                new_key = key.replace('icv_encoder.clip_embed.', '')
                if new_key not in icv_weights:
                    new_state_dict[new_key] = value
            self.clip_embed.load_state_dict(new_state_dict, strict=True)
        else:
            pass
                
    def feature_extractor(self, text_feature, image_feature,mode):
        if mode==1: ## Mean Aggregation
            return torch.mean(
                            torch.mean(text_feature,dim=0,keepdim=True)+torch.mean(image_feature,dim=0,keepdim=True),
                            dim=0,keepdim=True)
                        
        elif mode==2: ## Weighted Average
            image_features = torch.mean(image_feature * self.image_weight.transpose(0,1), dim=0, keepdim=True)
            text_features = torch.mean(text_feature * self.text_weight.transpose(0,1), dim=0, keepdim=True)
            return torch.mean(image_features+text_features,dim=0,keepdim=True)
        
        elif mode==3:##PCA
            image_pca_result = self.pca.fit_transform(image_feature)
            image_pca_vector = image_pca_result.components_[0]
            text_pca_result = self.pca.fit_transform(text_feature)
            text_pca_vector = text_pca_result.components_[0]
            return torch.mean(image_pca_vector,text_pca_vector,dim=0,keepdim=True)
        
        elif mode==4:##contact
            return torch.cat([torch.mean(image_feature,dim=0,keepdim=True),torch.mean(text_feature,dim=0,keepdim=True)],dim=-1)