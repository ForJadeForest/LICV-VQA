import os
import datasets
import json
from pathlib import Path


def load_vqav2_ds(
    vqav2_root_dir,
    train_coco_dataset_root,
    val_coco_dataset_root,
    split=None,
    val_ann_file=None,
):
    vqav2_root_dir = Path(vqav2_root_dir)
    train_ann = vqav2_root_dir / "v2_mscoco_train2014_annotations.json"
    train_ques = vqav2_root_dir / "v2_OpenEnded_mscoco_train2014_questions.json"
    if val_ann_file is not None:
        val_ann = vqav2_root_dir / val_ann_file
    else:
        val_ann = vqav2_root_dir / "v2_mscoco_val2014_annotations.json"
    val_ques = vqav2_root_dir / "v2_OpenEnded_mscoco_val2014_questions.json"

    def preprocess(ann_file, ques_file):
        ann = json.load(open(ann_file))["annotations"]
        ques = json.load(open(ques_file))["questions"]
        quesid2question = {}
        for q in ques:
            quesid2question[q["question_id"]] = q["question"]
        total_data = []
        for a in ann:
            a["question"] = quesid2question[a["question_id"]]
            total_data.append(a)
        return total_data

    if split == "train":
        train_ds_list = preprocess(train_ann, train_ques)
        ds = datasets.Dataset.from_list(train_ds_list)
    elif split == "validation":
        val_ds_list = preprocess(val_ann, val_ques)
        ds = datasets.Dataset.from_list(val_ds_list)
    else:
        train_ds_list = preprocess(train_ann, train_ques)
        val_ds_list = preprocess(val_ann, val_ques)
        ds = datasets.DatasetDict(
            {
                "train": datasets.Dataset.from_list(train_ds_list),
                "validation": datasets.Dataset.from_list(val_ds_list),
            }
        )

    ds = ds.sort("question_id")

    def train_trans(x, idx):
        filename = [f"COCO_train2014_{idx:012d}.jpg" for idx in x["image_id"]]
        img_path = [os.path.join(train_coco_dataset_root, f_n) for f_n in filename]

        x["image"] = img_path
        x["idx"] = idx
        return x

    def val_trans(x, idx):
        filename = [f"COCO_val2014_{idx:012d}.jpg" for idx in x["image_id"]]
        img_path = [os.path.join(val_coco_dataset_root, f_n) for f_n in filename]
        x["image"] = img_path
        x["idx"] = idx
        return x

    if split is None:
        ds["train"] = ds["train"].map(
            train_trans, batched=True, with_indices=True, num_proc=12
        )
        ds["validation"] = ds["validation"].map(
            val_trans, batched=True, with_indices=True, num_proc=12
        )
    elif split == "train":
        ds = ds.map(train_trans, batched=True, with_indices=True, num_proc=12)
    elif split == "validation":
        ds = ds.map(val_trans, batched=True, with_indices=True, num_proc=12)
    ds = ds.cast_column("image", datasets.Image(decode=True))

    ds = ds.rename_columns({"multiple_choice_answer": "answer"})

    return ds


def load_okvqa_ds(
    root_dir,
    train_coco_dataset_root,
    val_coco_dataset_root,
    split=None,
):
    root_dir = Path(root_dir)
    train_ann = root_dir / "mscoco_train2014_annotations.json"
    train_ques = root_dir / "OpenEnded_mscoco_train2014_questions.json"
    val_ann = root_dir / "mscoco_val2014_annotations.json"
    val_ques = root_dir / "OpenEnded_mscoco_val2014_questions.json"

    def preprocess(ann_file, ques_file):
        ann = json.load(open(ann_file))["annotations"]
        ques = json.load(open(ques_file))["questions"]
        quesid2question = {}
        for q in ques:
            quesid2question[q["question_id"]] = q["question"]
        total_data = []
        for a in ann:
            a["question"] = quesid2question[a["question_id"]]
            total_data.append(a)
        return total_data

    if split == "train":
        train_ds_list = preprocess(train_ann, train_ques)
        ds = datasets.Dataset.from_list(train_ds_list)
    elif split == "validation":
        val_ds_list = preprocess(val_ann, val_ques)
        ds = datasets.Dataset.from_list(val_ds_list)
    else:
        train_ds_list = preprocess(train_ann, train_ques)
        val_ds_list = preprocess(val_ann, val_ques)
        ds = datasets.DatasetDict(
            {
                "train": datasets.Dataset.from_list(train_ds_list),
                "validation": datasets.Dataset.from_list(val_ds_list),
            }
        )

    ds = ds.sort("question_id")

    def train_trans(x, idx):
        filename = [f"COCO_train2014_{idx:012d}.jpg" for idx in x["image_id"]]
        img_path = [os.path.join(train_coco_dataset_root, f_n) for f_n in filename]

        x["image"] = img_path
        x["idx"] = idx
        x["answer"] = [a[0]["answer"] for a in x["answers"]]
        return x

    def val_trans(x, idx):
        filename = [f"COCO_val2014_{idx:012d}.jpg" for idx in x["image_id"]]
        img_path = [os.path.join(val_coco_dataset_root, f_n) for f_n in filename]
        x["image"] = img_path
        x["idx"] = idx
        x["answer"] = [a[0]["answer"] for a in x["answers"]]
        return x

    if split is None:
        ds["train"] = ds["train"].map(
            train_trans, batched=True, with_indices=True, num_proc=12
        )
        ds["validation"] = ds["validation"].map(
            val_trans, batched=True, with_indices=True, num_proc=12
        )
    elif split == "train":
        ds = ds.map(train_trans, batched=True, with_indices=True, num_proc=12)
    elif split == "validation":
        ds = ds.map(val_trans, batched=True, with_indices=True, num_proc=12)
    ds = ds.cast_column("image", datasets.Image(decode=True))

    return ds
