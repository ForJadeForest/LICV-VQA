import re
from contextlib import nullcontext
from typing import List, Union
from loguru import logger

import torch.nn as nn
from baukit import TraceDict


class LearnableICVInterventionLMM(nn.Module):
    def __init__(
        self,
        lmm: nn.Module,
        enable_intervention=True,
        intervention_layer: Union[int, List[int]] = None,
        layer_format: str = None,
        total_layers: int = None,
    ):
        super().__init__()
        self.lmm = lmm

        if enable_intervention:
            self.total_layers = total_layers
            self.intervention_layers = self._prepare_layers(intervention_layer)
            self.intervention_layer_names = [
                layer_format.replace("<LAYER_NUM>", str(layer))
                for layer in self.intervention_layers
            ]
            logger.info(
                f"The intervention_layer_names is {self.intervention_layer_names}"
            )
            self.layer_to_icv_index = {
                int(layer_id): int(icv_idx)
                for icv_idx, layer_id in enumerate(self.intervention_layers)
            }
            logger.info(f"The layer_to_icv_index is {self.layer_to_icv_index}")
            self.intervention_enabled = True

    def _prepare_layers(self, layers):
        if layers == -1:
            return list(range(self.total_layers))
        return [layers] if isinstance(layers, int) else layers

    @property
    def device(self):
        return self.lmm.device

    @property
    def intervention_status(self) -> bool:
        return self.intervention_enabled

    @intervention_status.setter
    def intervention_status(self, value: bool):
        if not isinstance(value, bool):
            raise ValueError("Intervention status must be a boolean value.")
        self.intervention_enabled = value

    def toggle_intervention(self, enable: bool):
        self.intervention_status = enable

    def apply_icv_intervention(self, edit_layers, icv):
        def intervention_function(output, layer_name):
            layer_idx = int(re.findall(r"\d+", layer_name)[0])
            if layer_name in edit_layers and isinstance(output, tuple):
                hidden_states, *rest = output
                shift = icv[:, self.layer_to_icv_index[layer_idx]].unsqueeze(dim=1)
                shifted_states = hidden_states + shift
                normalized_states = (
                    shifted_states
                    / shifted_states.norm(dim=-1, keepdim=True)
                    * hidden_states.norm(dim=-1, keepdim=True)
                )
                return (normalized_states,) + tuple(rest)
            elif layer_name in edit_layers:
                hidden_states = output
                shift = icv[:, self.layer_to_icv_index[layer_idx]].unsqueeze(dim=1)
                shifted_states = hidden_states + shift
                normalized_states = (
                    shifted_states
                    / shifted_states.norm(dim=-1, keepdim=True)
                    * hidden_states.norm(dim=-1, keepdim=True)
                )
                return normalized_states
            return output

        return intervention_function

    def _get_context_manager(self, icv=None, retain_grad=False):
        if self.intervention_enabled:
            return TraceDict(
                self.lmm,
                layers=self.intervention_layer_names,
                edit_output=self.apply_icv_intervention(
                    self.intervention_layer_names, icv
                ),
                retain_grad=retain_grad,
            )
        return nullcontext()

    def forward(self, icv=None, *args, **kwargs):
        """
        Forward pass of the model with optional ICV intervention.

        Args:
            icv: Input control variable.
            *args: Variable length argument list of lmm.forward().
            **kwargs: Arbitrary keyword arguments of lmm.forward().

        Returns:
            The output of the model's forward pass.
        """
        with self._get_context_manager(icv, retain_grad=True):
            return self.lmm(*args, **kwargs)

    def generate(self, icv=None, *args, **kwargs):
        """
        Generate output using the specified ICV model with optional ICV intervention.

        Parameters:
            icv (ICVModel): The ICV model to use for generation.
            *args: Variable length argument list of lmm.generate().
            **kwargs: Arbitrary keyword arguments of lmm.generate().

        Returns:
            The generated output.

        """
        with self._get_context_manager(icv, retain_grad=False):
            return self.lmm.generate(*args, **kwargs)
