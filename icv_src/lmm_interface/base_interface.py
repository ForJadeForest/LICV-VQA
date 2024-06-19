from io import BytesIO
from typing import List

import requests
import torch
from loguru import logger
from openicl import PromptTemplate
from PIL import Image

from .utils import cast_type, get_autocast, is_url


class BaseInterface:
    def __init__(
        self,
        precision,
        device,
        input_ids_field_name,
        prompt_template,
        column_token_map,
        instruction,
        icd_join_char,
        label_field,
    ) -> None:
        self.data_type = cast_type(precision)
        self.autocast_context = get_autocast(precision)
        self.device = device
        self.input_ids_field_name = input_ids_field_name
        if not isinstance(prompt_template, str):
            prompt_template = dict(prompt_template)
        self.pt = PromptTemplate(
            prompt_template,
            column_token_map=dict(column_token_map),
        )
        self.icd_join_char = icd_join_char
        self.instruction = instruction
        self.pad_token_id = None
        self.tokenizer = None
        self.label_field = label_field

    def gen_text_with_label(self, item, label=None):
        if label is None and isinstance(self.pt.template, dict):
            label = item[self.label_field]

        return self.pt.generate_ice_item(item, label)

    def gen_text_without_label(self, item):
        return self.pt.generate_item(item, output_field=self.label_field)

    def concat_prompt(self, *args, **kwargs):
        raise NotImplemented

    def prepare_input(self, *args, **kwargs):
        raise NotImplemented

    @torch.inference_mode()
    def get_cond_prob(
        self,
        model_input,
        mask_length=None,
    ):
        ce_loss = self.get_ppl(model_input, mask_length)
        return (-ce_loss).exp()

    @torch.inference_mode()
    def get_ppl(
        self,
        model_input,
        mask_length=None,
    ):
        if self.pad_token_id is None:
            logger.warning("the pad_token_id is None")
        with self.autocast_context:
            outputs = self.model(**model_input)

            shift_logits = outputs.logits[..., :-1, :].contiguous()
            shift_labels = model_input[self.input_ids_field_name][..., 1:].contiguous()
            loss_fct = torch.nn.CrossEntropyLoss(
                reduction="none", ignore_index=self.pad_token_id
            )
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
            )
            loss = loss.view(shift_labels.size())

            if mask_length is not None:
                loss_mask = torch.zeros_like(shift_labels)  # [batch, seqlen]
                for i in range(len(loss_mask)):
                    for j in range(mask_length[i] - 1, len(loss_mask[i])):
                        loss_mask[i][j] = 1
                loss = loss * loss_mask
            lens = (model_input[self.input_ids_field_name] != self.pad_token_id).sum(-1)

            if mask_length is not None:
                lens -= torch.tensor(mask_length, device=lens.device)
            # logger.debug(f'{lens=}')
            ce_loss = loss.sum(-1) / lens
        return ce_loss

    def get_input_token_num(self, input_tokens: str) -> int:
        return len(self.tokenizer(input_tokens, add_special_tokens=False)["input_ids"])

    def transfer_prompts(
        self, batch_data_sample_list, is_last_for_generation=True, query_label=None
    ):
        raise NotImplemented

    def generate(self, *args, **kwargs):
        with self.autocast_context:
            return self.model.generate(*args, **kwargs)

    def __call__(self, model_input):
        with self.autocast_context:
            return self.model(**model_input)


class LMMInterface(BaseInterface):
    def __init__(
        self,
        precision,
        device,
        input_ids_field_name,
        prompt_template,
        column_token_map,
        instruction,
        icd_join_char,
        label_field,
        image_field,
    ):
        super().__init__(
            precision,
            device,
            input_ids_field_name,
            prompt_template,
            column_token_map,
            instruction,
            icd_join_char,
            label_field,
        )
        self.image_field = image_field

    def is_img(self, obj):
        if isinstance(obj, Image.Image):
            return obj
        elif isinstance(obj, str):
            if is_url(obj):
                headers = {
                    "User-Agent": (
                        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0"
                        " Safari/537.36"
                    )
                }
                response = requests.get(obj, stream=True, headers=headers)
                response.raise_for_status()
                return Image.open(BytesIO(response.content))
            else:
                try:
                    return Image.open(obj)
                except:
                    return None

    def gen_text_with_label(self, item, label=None, add_image_token=False):
        prompt = super().gen_text_with_label(item, label)
        if add_image_token:
            return self.add_image_token(prompt)
        return prompt

    def gen_text_without_label(self, item, add_image_token=False):
        prompt = self.pt.generate_item(item, output_field=self.label_field)
        if add_image_token:
            return self.add_image_token(prompt)
        return prompt

    def transfer_prompts(
        self, batch_data_sample_list, is_last_for_generation=True, query_label=None
    ):
        """
        transfer data sample list to text input.
        Note: Only support one image and one text pair.
        """
        if not any(isinstance(i, list) for i in batch_data_sample_list):
            batch_data_sample_list = [batch_data_sample_list]

        prompts = []
        for data_sample_list in batch_data_sample_list:
            prompt = []
            for data_sample in data_sample_list[:-1]:
                prompt.extend(
                    [
                        data_sample[self.image_field],
                        self.gen_text_with_label(data_sample),
                    ]
                )
            prompt.append(data_sample_list[-1][self.image_field])
            if is_last_for_generation:
                prompt.append(self.gen_text_without_label(data_sample_list[-1]))
            else:
                prompt.append(
                    self.gen_text_with_label(data_sample_list[-1], label=query_label)
                )

            prompts.append(prompt)
        return prompts

    def add_image_token(self, text):
        raise NotImplemented
