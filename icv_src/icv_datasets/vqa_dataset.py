from torch.utils.data import Dataset
import numpy as np
from .load_ds_utils import load_okvqa_ds, load_vqav2_ds
import os
import datasets
import json
from pathlib import Path
import torch
from tqdm import tqdm

class VQAV2Dataset(Dataset):
    def __init__(
        self,
        name,
        root_dir,
        train_coco_dataset_root,
        val_coco_dataset_root,
        instruction="",
        few_shot_num=8,
        max_train_size=10000,
        split="train",
        retriver="random",
        faiss_directory=None,
        val_ann_file=None,
    ):
        super().__init__()
        if name == "vqav2":
            self.prime_ds = load_vqav2_ds(
                root_dir,
                train_coco_dataset_root,
                val_coco_dataset_root,
                split=split,
                val_ann_file=val_ann_file,
            )
        elif name == "okvqa":
            self.prime_ds = load_okvqa_ds(
                root_dir,
                train_coco_dataset_root,
                val_coco_dataset_root,
                split=split,
            )
        self.retriver = retriver
        self.fassis_directory = faiss_directory
        self.name = name
        print("the current retriver is", self.retriver)
        print("_____")
        if self.retriver=="load_dictonary":
                print("loading dictonary...")
                with open(self.fassis_directory+ f"/{self.name}.json", "r") as f:
                        dictonary = json.load(f)
                random_select_idx = [int(list(i.keys())[0]) for i in dictonary]
                self.ds = self.prime_ds.select(random_select_idx)
                self.dic = {key: value for d in dictonary for  key, value in d.items()}
                print("loaded")
        else:
                print("random")
                random_select_idx = np.random.randint(0, len(self.prime_ds), size=max_train_size)
                dictonary = []
                self.ds = self.prime_ds.select(random_select_idx)
                if max_train_size > 0 and len(self.ds) > max_train_size:
                    random_select_idx = np.random.randint(0, len(self.ds), size=max_train_size)
                    self.ds = self.ds.select(random_select_idx)
        
        self.few_shot_num = few_shot_num
        self.instruction = instruction
        
        

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, index):
        batch_size = 2048
        
        def process_batch(batch_images):
            inputs = processor(images=batch_images, return_tensors="pt", padding=True, return_attention_mask=False).to(device)
            with torch.no_grad():
                image_features = model.get_image_features(**inputs)
            return image_features.cpu().detach().numpy()
        query_item = self.ds[index]
        if self.retriver == "random":
            #print("the current retriver is random")
            few_shot_index = np.random.randint(0, len(self.prime_ds), size=self.few_shot_num)
            few_shot_index = few_shot_index.tolist()
        elif self.retriver == "siir":
            #print("the current retriver is siir")
            device = "cuda" 
            from transformers import CLIPProcessor, CLIPModel
            model = CLIPModel.from_pretrained("/data1/pyz/model_weight/clip-vit-base-patch32/").to(device)
            processor = CLIPProcessor.from_pretrained("/data1/pyz/model_weight/clip-vit-base-patch32/")
            if os.path.exists(self.fassis_directory) == False:
            #如果不存在fassis_directory，则创建
                with torch.no_grad():
                    batch_images = []
                    image_index = faiss.IndexFlatL2(512)
                    for idx in tqdm(range(len(self.prime_ds))):
                        image = self.prime_ds[idx]["image"]
                        batch_images.append(image)
                        if len(batch_images) == batch_size or idx == len(self.prime_ds) - 1:
                            image_features = process_batch(batch_images)
                            image_index.add(image_features)
                            batch_images = []  

                    faiss.write_index(image_index, self.fassis_directory + f"{self.name}.faiss")
            else:
                image_index = faiss.read_index(self.fassis_directory+ f"{self.name}.faiss")
                query_image = query_item["image"]
                query_image =processor(images=query_image, return_tensors="pt").to(device)
                query_image_feature = model.get_image_features(**query_image)
                query_image_feature = query_image_feature.cpu().detach().numpy()  
                D, I = image_index.search(query_image_feature, self.few_shot_num+2)
                if index in I[0]:
                    I[0] = np.delete(I[0], np.where(I[0] == index))
                    few_shot_index = I[0][0:self.few_shot_num]
                else:
                    few_shot_index = I[0][0:self.few_shot_num]
        elif self.retriver == "load_dictonary":
            #print("the current retriver is load_dictonary")
            #load an exisiting json which contains ice ids and query id to build ice
            query_ready_index = self.ds[index]["idx"]
            few_shot_index=self.dic[str(query_ready_index)]
        #print(few_shot_index)
        in_context_example = [self.prime_ds[idx] for idx in few_shot_index]

        prompt = []
        if self.instruction:
            prompt = [self.instruction]
        for ice in in_context_example:
            prompt.append(ice["image"])
            prompt.append(f"Question:{ice['question']} Short answer:{ice['answer']}\n")

        query_prompt = [
            query_item["image"],
            f"Question:{query_item['question']} Short answer:{query_item['answer']}",
        ]

        query_x = [ice["image"], f"Question:{query_item['question']} Short answer:"]
        #print("data:",query_x)
        return {
            "ice_prompt": prompt,
            "query_prompt": query_prompt,
            "query_x": query_x,
        }
