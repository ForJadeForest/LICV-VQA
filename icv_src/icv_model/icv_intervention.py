from typing import Union, List
import torch.nn as nn
from baukit import TraceDict
from contextlib import nullcontext


class LearnableICVInterventionLMM(nn.Module):
    def __init__(
        self,
        lmm: nn.Module,
        intervention_layer: Union[int, List[int]],
        layer_format: str,
        total_layers: int,
    ):
        super().__init__()
        self.lmm = lmm
        self.total_layers = total_layers
        self.intervention_layers = self._prepare_layers(intervention_layer)
        self.intervention_layer_names = [
            layer_format.replace("<LAYER_NUM>", str(layer))
            for layer in self.intervention_layers
        ]
        self.layer_to_icv_index = {
            layer_id: icv_idx
            for layer_id, icv_idx in enumerate(self.intervention_layers)
        }
        self.intervention_enabled = True

    def _prepare_layers(self, layers):
        if layers == -1:
            return list(range(self.total_layers))
        return [layers] if isinstance(layers, int) else layers

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
            layer_idx = int(layer_name.split(".")[2])
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

    def forward(self, input_data, icv=None):
        with self._get_context_manager(icv, retain_grad=True):
            return self.lmm(**input_data)

    def generate(self, input_data, generation_args=None, icv=None):
        if generation_args is None:
            generation_args = {}

        with self._get_context_manager(icv, retain_grad=False):
            return self.lmm.generate(**input_data, **generation_args)
