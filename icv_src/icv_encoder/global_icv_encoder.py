from .base_icv_encoder import BaseICVEncoder, ICVEncoderOutput
import torch


class GlobalICVEncoder(BaseICVEncoder):
    def __init__(
        self,
        lmm_hidden_dim,
        lmm_layers,
        alpha_learnable=True,
        alpha_init_value=0.0,
        use_sigmoid=False,
    ) -> None:
        """
        Initializes the GlobalICVEncoder object.

        Args:
            lmm_hidden_dim (int): The hidden dimension of the LMM layers.
            lmm_layers (int): The number of LMM layers.
            alpha_learnable (bool, optional): Whether the alpha parameter is learnable. Defaults to True.
            alpha_init_value (float, optional): The initial value of the alpha parameter. Defaults to 0.0.
            use_sigmoid (bool, optional): Whether to use sigmoid activation. Defaults to False.
        """
        super().__init__()

        self.alpha = torch.nn.Parameter(
            torch.full(size=(1, lmm_layers), fill_value=float(alpha_init_value)),
            requires_grad=alpha_learnable,
        )
        self.icv = torch.nn.Parameter(torch.empty(1, lmm_layers, lmm_hidden_dim))
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
