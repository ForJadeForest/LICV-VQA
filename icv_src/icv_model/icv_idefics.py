""" PyTorch Idefics model."""

from typing import List, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers.activations import ACT2FN
from transformers.modeling_attn_mask_utils import (
    _prepare_4d_causal_attention_mask_for_sdpa,
)
from transformers.models.idefics.configuration_idefics import IdeficsConfig
from transformers.models.idefics.modeling_idefics import (
    LLAMA_INPUTS_DOCSTRING,
    LLAMA_START_DOCSTRING,
    IdeficsBaseModelOutputWithPast,
    IdeficsCausalLMOutputWithPast,
    IdeficsDecoderLayer,
    IdeficsDecoupledEmbedding,
    IdeficsDecoupledLinear,
    IdeficsGatedCrossAttentionLayer,
    IdeficsPreTrainedModel,
    IdeficsRMSNorm,
    expand_inputs_for_generation,
    freeze_model,
    prepare_inputs_for_generation,
    update_model_kwargs_for_generation,
)
from transformers.models.idefics.perceiver import IdeficsPerceiverResampler
from transformers.models.idefics.vision import IdeficsVisionTransformer
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
from transformers.utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)

logger = logging.get_logger(__name__)
_CONFIG_FOR_DOC = "IdeficsConfig"


@add_start_docstrings(
    "The bare LLaMA Model outputting raw hidden-states without any specific head on top.",
    LLAMA_START_DOCSTRING,
)
class ICVIdeficsModel(IdeficsPreTrainedModel):
    """
    Transformer decoder consisting of `config.num_hidden_layers` layers. Each layer is a [`IdeficsDecoderLayer`]

    Args:
        config: IdeficsConfig
    """

    def __init__(self, config: IdeficsConfig):
        super().__init__(config)
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = IdeficsDecoupledEmbedding(
            num_embeddings=config.vocab_size,
            num_additional_embeddings=config.additional_vocab_size,
            embedding_dim=config.hidden_size,
            partially_freeze=config.freeze_text_layers,
            padding_idx=self.padding_idx,
        )

        self.image_size = config.vision_config.image_size
        self.vision_config = config.vision_config
        self.vision_model = IdeficsVisionTransformer(config.vision_config)

        # Perceiver Resampler
        if config.use_resampler:
            perceiver_config = config.perceiver_config
            self.perceiver_resampler = IdeficsPerceiverResampler(
                config,
                config.vision_config.embed_dim,
                perceiver_config.resampler_depth,
                perceiver_config.resampler_n_heads,
                perceiver_config.resampler_head_dim,
                perceiver_config.resampler_n_latents,
            )

        self.layers = nn.ModuleList(
            [IdeficsDecoderLayer(config) for _ in range(config.num_hidden_layers)]
        )

        self.cross_layer_interval = config.cross_layer_interval
        num_cross_layers = config.num_hidden_layers // self.cross_layer_interval
        self.gated_cross_attn_layers = nn.ModuleList(
            [IdeficsGatedCrossAttentionLayer(config) for _ in range(num_cross_layers)]
        )
        self.gradient_checkpointing = False

        self.norm = IdeficsRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # Initialize weights and apply final processing
        self.post_init()

        self.freeze_relevant_params(config)

    def freeze_relevant_params(self, config=None):
        if config is None:
            config = self.config

        if config.freeze_text_layers:
            self.freeze_text_layers(config.freeze_text_module_exceptions)

        if config.freeze_vision_layers:
            freeze_model(
                self.vision_model,
                module_exceptions=config.freeze_vision_module_exceptions,
            )

    def freeze_text_layers(self, module_exceptions=[]):
        for module in [self.layers, self.norm]:
            freeze_model(module, module_exceptions=module_exceptions)

    def freeze_vision_layers(self, module_exceptions=[]):
        freeze_model(self.vision_model, module_exceptions=module_exceptions)

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        image_encoder_embeddings: Optional[torch.FloatTensor] = None,
        perceiver_embeddings: Optional[torch.FloatTensor] = None,
        image_attention_mask: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        interpolate_pos_encoding: Optional[bool] = False,
        return_dict: Optional[bool] = None,
        in_context_vector: Optional[torch.FloatTensor] = None,
        alpha: Optional[torch.FloatTensor] = None,
    ) -> Union[Tuple, IdeficsBaseModelOutputWithPast]:
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time"
            )
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError(
                "You have to specify either decoder_input_ids or decoder_inputs_embeds"
            )

        seq_length_with_past = seq_length
        past_key_values_length = 0

        if past_key_values is not None:
            past_key_values_length = past_key_values[0][0].shape[2]
            seq_length_with_past = seq_length_with_past + past_key_values_length

        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
        elif position_ids is None:
            position_ids = torch.arange(
                past_key_values_length,
                seq_length + past_key_values_length,
                dtype=torch.long,
                device=device,
            )
            position_ids = position_ids.unsqueeze(0)

        if (pixel_values, image_encoder_embeddings, perceiver_embeddings).count(
            None
        ) != 2:
            raise ValueError(
                "Exactly 1 of pixel_values, image_encoder_embeddings or perceiver_embeddings has to be not-None."
            )

        elif pixel_values is not None:
            pixel_values = pixel_values.to(
                dtype=self.dtype, device=device
            )  # fp16 compatibility
            batch_size, num_images = pixel_values.shape[:2]
            pixel_values = pixel_values.contiguous().view(
                batch_size * num_images, *pixel_values.shape[2:]
            )

            # Get sequence from the vision encoder
            image_hidden_states = self.vision_model(
                pixel_values=pixel_values,
                interpolate_pos_encoding=interpolate_pos_encoding,
            ).last_hidden_state

        elif image_encoder_embeddings is not None:
            batch_size, num_images, image_seq_len, image_hidden_size = (
                image_encoder_embeddings.size()
            )
            image_hidden_states = image_encoder_embeddings.to(
                dtype=self.dtype, device=device
            )
            image_hidden_states = image_hidden_states.view(
                batch_size * num_images, image_seq_len, image_hidden_size
            )

        if self.config.use_resampler:
            if perceiver_embeddings is None:
                perceiver_embeddings = self.perceiver_resampler(image_hidden_states)
                image_seq_len, image_hidden_size = perceiver_embeddings.size(
                    1
                ), perceiver_embeddings.size(2)
            else:
                batch_size, num_images, image_seq_len, image_hidden_size = (
                    perceiver_embeddings.size()
                )
            image_hidden_states = perceiver_embeddings
        elif perceiver_embeddings is None:
            image_seq_len, image_hidden_size = image_hidden_states.size(
                1
            ), image_hidden_states.size(2)
        else:
            raise ValueError(
                "If `perceiver_embeddings` are passed, use_resampler should be True"
            )

        image_hidden_states = image_hidden_states.view(
            batch_size, num_images * image_seq_len, image_hidden_size
        )
        # # Hack to use the model in full language modeling mode
        # image_attention_mask = torch.zeros(batch_size, seq_length, 1, dtype=torch.long, device=image_hidden_states.device)
        # Make image_attention_mask compatible with hidden states
        text_seq_len = image_attention_mask.size(1)
        image_attention_mask = image_attention_mask.unsqueeze(-1)
        image_attention_mask = image_attention_mask.repeat(1, 1, 1, image_seq_len)
        image_attention_mask = image_attention_mask.view(
            batch_size, text_seq_len, num_images * image_seq_len
        )

        if image_hidden_states is not None:
            image_batch_size, image_sequence_length, _ = image_hidden_states.size()
            image_hidden_shape = (image_batch_size, image_sequence_length)
            if image_attention_mask is None:
                image_attention_mask = torch.ones(image_hidden_shape, device=device)
            image_attention_mask = self.invert_attention_mask(image_attention_mask)
        else:
            image_attention_mask = None

        # cross_attention_gate:
        # For any tokens attending to no images, the hidden_states comming out of the cross-attention should be zeroed-out.
        # `image_attention_mask` has shape [bsz, 1, num_images, hidden_size] with elements equal to either 0.0 or a very negative number.
        # If any of the elements are 0.0, then the token is attending to at least one image and the gate value is 1. Otherwise the gate value is 0.
        # `cross_attention_gate` has shape [bsz, seq_len] with elements equal to either 0.0 or 1.0.
        cross_attention_gate = (
            (((image_attention_mask == 0.0).any(dim=-1)).to(dtype=self.dtype)).squeeze(
                dim=1
            )
        ).to(device)

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        # embed positions
        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, seq_length_with_past),
                dtype=torch.bool,
                device=inputs_embeds.device,
            )
        attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
            attention_mask,
            (batch_size, seq_length),
            inputs_embeds,
            past_key_values_length,
        )

        hidden_states = inputs_embeds

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None

        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            past_key_value = (
                past_key_values[idx] if past_key_values is not None else None
            )

            def vblock(
                main_block,
                hidden_states,
                attention_mask,
                position_ids,
                past_key_value,
                image_hidden_states,
                image_attention_mask,
                cross_attention_gate,
                output_attentions,
                use_cache,
                layer_idx,
                cross_layer_interval,
                gated_cross_attn_layers,
            ):
                # TODO(ls): Add cross attention values to respective lists
                if layer_idx % cross_layer_interval == 0:
                    xblock = gated_cross_attn_layers[layer_idx // cross_layer_interval]
                    outputs = xblock(
                        hidden_states,
                        attention_mask=attention_mask,
                        image_hidden_states=image_hidden_states,
                        image_attention_mask=image_attention_mask,
                        cross_attention_gate=cross_attention_gate,
                        output_attentions=output_attentions,
                        use_cache=use_cache,
                        past_key_value=None,  # not implemented
                    )
                    hidden_states = outputs[0]

                layer_outputs = main_block(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )

                return layer_outputs

            if self.gradient_checkpointing and self.training:
                past_key_value = None
                if use_cache:
                    logger.warning_once(
                        "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                    )
                    use_cache = False

                layer_outputs = self._gradient_checkpointing_func(
                    vblock,
                    decoder_layer,
                    hidden_states,
                    attention_mask,
                    position_ids,
                    past_key_value,
                    image_hidden_states,
                    image_attention_mask,
                    cross_attention_gate,
                    output_attentions,
                    use_cache,
                    idx,
                    self.cross_layer_interval,
                    self.gated_cross_attn_layers,
                )
            else:
                layer_outputs = vblock(
                    decoder_layer,
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_value,
                    image_hidden_states=image_hidden_states,
                    image_attention_mask=image_attention_mask,
                    cross_attention_gate=cross_attention_gate,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    layer_idx=idx,
                    cross_layer_interval=self.cross_layer_interval,
                    gated_cross_attn_layers=self.gated_cross_attn_layers,
                )

            hidden_states = layer_outputs[0]
            if in_context_vector is not None:
                shift_dis = alpha[:, idx].unsqueeze(dim=1) * in_context_vector[:, idx]
                shift_hidden_states = hidden_states + shift_dis.unsqueeze(dim=1)
                hidden_states = (
                    shift_hidden_states
                    / shift_hidden_states.norm(dim=-1, keepdim=True)
                    * hidden_states.norm(dim=-1, keepdim=True)
                )
            if use_cache:
                next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        image_hidden_states = image_hidden_states.view(
            batch_size, num_images, image_seq_len, image_hidden_size
        )
        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    next_cache,
                    all_hidden_states,
                    all_self_attns,
                    image_hidden_states,
                ]
                if v is not None
            )
        return IdeficsBaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            image_hidden_states=image_hidden_states,
        )


class ICVIdeficsForVisionText2Text(IdeficsPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"lm_head.weight"]
    _tied_weights_keys = ["model.embed_tokens.weight", "lm_head.weight"]

    def __init__(self, config, vision_model=None):
        super().__init__(config)
        self.model = ICVIdeficsModel(config)

        self.lm_head = IdeficsDecoupledLinear(
            in_features=config.hidden_size,
            out_features=config.vocab_size,
            out_additional_features=config.additional_vocab_size,
            bias=False,
            partially_freeze=config.freeze_lm_head,
        )

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def tie_weights(self):
        """
        Overwrite `transformers.modeling_utils.PreTrainedModel.tie_weights` to handle the case of
        IdeficsDecoupledLinear and IdeficsDecoupledEmbedding.
        """
        output_embeddings = self.get_output_embeddings()
        input_embeddings = self.get_input_embeddings()

        if getattr(self.config, "tie_word_embeddings", True):
            output_embeddings.weight = input_embeddings.weight
            if input_embeddings.num_additional_embeddings > 0:
                assert (
                    output_embeddings.out_additional_features
                    == input_embeddings.num_additional_embeddings
                )
                output_embeddings.additional_fc.weight = (
                    input_embeddings.additional_embedding.weight
                )

        if hasattr(output_embeddings, "out_features") and hasattr(
            input_embeddings, "num_embeddings"
        ):
            output_embeddings.out_features = input_embeddings.num_embeddings
            if hasattr(output_embeddings, "out_additional_features") and hasattr(
                input_embeddings, "num_additional_embeddings"
            ):
                output_embeddings.out_additional_features = (
                    input_embeddings.num_additional_embeddings
                )

    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    @replace_return_docstrings(
        output_type=IdeficsCausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC
    )
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        image_encoder_embeddings: Optional[torch.FloatTensor] = None,
        perceiver_embeddings: Optional[torch.FloatTensor] = None,
        image_attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        interpolate_pos_encoding: Optional[bool] = False,
        return_dict: Optional[bool] = None,
        in_context_vector: Optional[torch.FloatTensor] = None,
        alpha: Optional[torch.FloatTensor] = None,
    ) -> Union[Tuple, IdeficsCausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, IdeficsForVisionText2Text

        >>> model = IdeficsForVisionText2Text.from_pretrained("HuggingFaceM4/idefics-9b")
        >>> tokenizer = AutoTokenizer.from_pretrained("HuggingFaceM4/idefics-9b")

        >>> prompt = "Hey, are you consciours? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you consciours? Can you talk to me?\nI'm not consciours, but I can talk to you."
        ```"""

        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            pixel_values=pixel_values,
            image_encoder_embeddings=image_encoder_embeddings,
            perceiver_embeddings=perceiver_embeddings,
            image_attention_mask=image_attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            interpolate_pos_encoding=interpolate_pos_encoding,
            return_dict=return_dict,
            in_context_vector=in_context_vector,
            alpha=alpha,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            labels = labels.to(logits.device)
            # Shift so that tokens < n predict n
            if attention_mask is not None:
                shift_attention_mask = attention_mask[..., 1:].to(logits.device)
                shift_logits = logits[..., :-1, :][
                    shift_attention_mask != 0
                ].contiguous()
                shift_labels = labels[..., 1:][shift_attention_mask != 0].contiguous()
            else:
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
            )

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return IdeficsCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            image_hidden_states=outputs.image_hidden_states,
        )

    def prepare_inputs_for_generation(self, input_ids, past=None, **kwargs):
        image_hidden_states = kwargs.pop("image_hidden_states", None)
        if image_hidden_states is not None:
            if self.config.use_resampler:
                kwargs["perceiver_embeddings"] = image_hidden_states
            else:
                kwargs["image_encoder_embeddings"] = image_hidden_states
            kwargs["pixel_values"] = None
        inputs = prepare_inputs_for_generation(input_ids, past=past, **kwargs)
        unwanted_kwargs = ["token_type_ids"]
        for kwarg in unwanted_kwargs:
            inputs.pop(kwarg, None)
        if "in_context_vector" in kwargs:
            inputs["in_context_vector"] = kwargs["in_context_vector"]
            inputs["alpha"] = kwargs["alpha"]
        return inputs

    @staticmethod
    def _expand_inputs_for_generation(
        *args,
        **model_kwargs,
    ):
        return expand_inputs_for_generation(*args, **model_kwargs)

    @staticmethod
    def _update_model_kwargs_for_generation(outputs, model_kwargs, is_encoder_decoder):
        return update_model_kwargs_for_generation(outputs, model_kwargs)

    @staticmethod
    def _reorder_cache(past, beam_idx):
        reordered_past = ()
        for layer_past in past:
            reordered_past += (
                tuple(
                    past_state.index_select(0, beam_idx) for past_state in layer_past
                ),
            )
        return reordered_past
