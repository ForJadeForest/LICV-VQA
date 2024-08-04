"""MultiModal Topk Retriever"""

import os
from typing import List, Optional

import datasets
import faiss
import numpy as np
import torch
import tqdm
from loguru import logger
from torch.utils.data import DataLoader
from transformers import (
    AutoProcessor,
    AutoTokenizer,
    CLIPTextModelWithProjection,
    CLIPVisionModelWithProjection,
)


class MMTopkRetriever:
    def __init__(
        self,
        index_ds: datasets.Dataset,
        test_ds: datasets.Dataset,
        clip_model_name: Optional[str] = "openai/clip-vit-base-patch32",
        mode: Optional[str] = "i2t",
        index_field: Optional[str] = "text",
        test_field: Optional[str] = "image",
        batch_size: Optional[int] = 1,
        num_workers: Optional[int] = 0,
        cache_file: Optional[str] = None,
        device: str = "cuda",
        reversed_order: Optional[bool] = False,
    ) -> None:
        """Initialize the MMTopkRetriever.

        Args:
            index_ds (datasets.Dataset): The dataset used for creating the index.
            test_ds (datasets.Dataset): The dataset used for testing.
            clip_model_name (Optional[str], optional): The name of the CLIP model. Defaults to 'openai/clip-vit-base-patch32'.
            mode (Optional[str], optional): The mode of operation, 'i2t' for image to text or 't2i' for text to image. Defaults to 'i2t'.
            index_field (Optional[str], optional): The field in the index dataset to use. Defaults to 'text'.
            test_field (Optional[str], optional): The field in the test dataset to use. Defaults to 'image'.
            batch_size (Optional[int], optional): The batch size for processing. Defaults to 1.
            num_workers (Optional[int], optional): The number of workers for data loading. Defaults to 0.
            cache_file (Optional[str], optional): Path to the cache file. Defaults to None.
            device (str, optional): The device to use for computations. Defaults to 'cuda'.
            reversed_order (Optional[bool], optional): Whether to reverse the order of retrieved results. Defaults to False.
        """
        self.index_ds = index_ds
        self.test_ds = test_ds

        self.clip_model_name = clip_model_name
        self.device = device
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.mode = mode
        self.index_field = index_field
        self.test_field = test_field
        self.reversed_order = reversed_order

        if cache_file is None or not os.path.exists(cache_file):
            self.create_index(cache_file)
        else:
            logger.info(f"cache_file: {cache_file} exist: begin loadding...")
            features = torch.load(cache_file)
            self.index_features = features["index_features"]
            self.test_features = features["test_features"]
            emb_dim = self.index_features.shape[1]
            self.index = faiss.IndexFlatIP(emb_dim)
            logger.info(f"begin add the index for emb dim: {self.index_features.shape}")
            self.index.add(self.index_features)

    def create_index(self, cache_file: Optional[str]) -> None:
        """Create the index for retrieval.

        Args:
            cache_file (Optional[str]): The path to the cache file. If None, the index is created from scratch.
        """
        logger.info(f"begin load {self.clip_model_name} text encodcer")
        self.text_encoder = CLIPTextModelWithProjection.from_pretrained(
            self.clip_model_name
        ).to(self.device)
        logger.info(f"begin load {self.clip_model_name} image encodcer")
        self.vision_encoder = CLIPVisionModelWithProjection.from_pretrained(
            self.clip_model_name
        ).to(self.device)

        self.text_encoder.eval()
        self.vision_encoder.eval()

        logger.info(f"begin load {self.clip_model_name} processor and tokenizer...")
        self.img_processor = AutoProcessor.from_pretrained(self.clip_model_name)
        self.tokenzier = AutoTokenizer.from_pretrained(self.clip_model_name)

        encoding_method_map = {"i": self.encode_img, "t": self.encode_text}
        index_encoding = self.mode.split("2")[1]
        test_encoding = self.mode.split("2")[0]

        self.index_features = encoding_method_map[index_encoding](
            self.index_ds, self.index_field
        )
        self.test_features = encoding_method_map[test_encoding](
            self.test_ds, self.test_field
        )

        cache_feature = {
            "index_features": self.index_features,
            "test_features": self.test_features,
            "meta_info": {
                "clip_model_name": self.clip_model_name,
                "mode": self.mode,
                "index_field": self.index_field,
                "test_field": self.test_field,
            },
        }
        torch.save(cache_feature, cache_file)
        emb_dim = self.index_features.shape[1]
        self.index = faiss.IndexFlatIP(emb_dim)

        logger.info(f"begin add the index for emb dim: {self.index_features.shape}")
        self.index.add(self.index_features)
        del self.text_encoder
        del self.vision_encoder

    @torch.inference_mode()
    def encode_text(self, ds: datasets.Dataset, text_field: str) -> np.ndarray:
        """Encode text from a given dataset.

        Args:
            ds (datasets.Dataset): The dataset containing the text to be encoded.
            text_field (str): The field in the dataset that contains the text data.

        Returns:
            np.ndarray: An array of encoded text features.
        """
        logger.info(f"now begin tokenizer field: {text_field}")
        remove_columns = ds.column_names

        text_ds = ds.map(
            lambda x: self.tokenzier(
                x[text_field], padding=True, return_tensors="pt", truncation=True
            ),
            batched=True,
            batch_size=self.batch_size,
            remove_columns=remove_columns,
        )
        text_ds.set_format(type="torch", columns=["input_ids", "attention_mask"])

        dataloader = DataLoader(text_ds, batch_size=self.batch_size, shuffle=False)
        logger.info(
            f"use {self.clip_model_name} to encode the text field: {text_field}"
        )
        bar = tqdm.tqdm(dataloader)

        feature_list = []
        for batch_data in bar:
            features = self.text_encoder(
                input_ids=batch_data["input_ids"].to(self.device),
                attention_mask=batch_data["attention_mask"].to(self.device),
            ).text_embeds
            features /= features.norm(dim=-1, keepdim=True)
            feature_list.append(features)
        features = torch.cat(feature_list, dim=0)
        return features.cpu().detach().numpy()

    @torch.inference_mode()
    def encode_img(self, ds: datasets.Dataset, img_field: str) -> np.ndarray:
        """
        Encodes images from a dataset into feature vectors using a pre-trained vision model.

        Args:
            ds (datasets.Dataset): The dataset containing the images to be encoded.
            img_field (str): The field name in the dataset where the image data is stored.

        Returns:
            np.ndarray: An array of encoded image features.
        """

        logger.info(f"now begin processor img field: {img_field}")

        ds_ = ds.map()
        ds_ = ds_.cast_column(img_field, datasets.Image(decode=True))

        def prepare(examples):
            images = [i for i in examples[img_field]]
            data_dict = {}

            data_dict["pixel_values"] = self.img_processor(
                images=images,
                return_tensors="pt",
            )["pixel_values"]
            return data_dict

        ds_.set_transform(prepare)

        dataloader = DataLoader(
            ds_, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers
        )
        logger.info(f"use {self.clip_model_name} to encode the img field: {img_field}")
        bar = tqdm.tqdm(dataloader)

        feature_list = []
        for batch_data in bar:
            features = self.vision_encoder(
                batch_data["pixel_values"].squeeze(dim=1).to(self.device)
            ).image_embeds
            features /= features.norm(dim=-1, keepdim=True)
            feature_list.append(features)
        features = torch.cat(feature_list, dim=0)
        return features.cpu().detach().numpy()

    def retrieve(self, ice_num: int) -> List[List[int]]:
        """Retrieve the top-k closest items from the index.

        Args:
            ice_num (int): The number of closest items to retrieve for each query.

        Returns:
            List[List[int]]: A list of lists, where each sublist contains the indices of the top-k closest items for a corresponding query.
        """
        idx_list = self.index.search(self.test_features, ice_num)[1].tolist()
        if self.reversed_order:
            idx_list = [list(reversed(i)) for i in idx_list]
        return idx_list
