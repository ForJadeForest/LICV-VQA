import os
import datasets
from datasets import DatasetDict
import json
from pathlib import Path


def load_vqav2_ds(
    root_dir,
    train_coco_dataset_root,
    val_coco_dataset_root,
    split=None,
    val_ann_file=None,
    filter_ques_type: str = None,
):
    root_dir = Path(root_dir)
    train_ann = root_dir / "v2_mscoco_train2014_annotations.json"
    train_ques = root_dir / "v2_OpenEnded_mscoco_train2014_questions.json"
    if val_ann_file is not None:
        val_ann = root_dir / val_ann_file
    else:
        val_ann = root_dir / "v2_mscoco_val2014_annotations.json"
    val_ques = root_dir / "v2_OpenEnded_mscoco_val2014_questions.json"

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

    def ques_type_gene(examples):
        examples["gen_question_type"] = [
            que_type.split(" ")[0] for que_type in examples["question_type"]
        ]
        return examples

    ds = ds.map(ques_type_gene, batched=True, num_proc=12)
    if filter_ques_type:
        ds = ds.filter(
            lambda x: [i == filter_ques_type for i in x["gen_question_type"]],
            batched=True,
        )

    return ds


def load_okvqa_ds(
    root_dir,
    train_coco_dataset_root,
    val_coco_dataset_root,
    split=None,
    filter_ques_type: str = None,
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

    def ques_type_gene(examples):
        examples["gen_question_type"] = [
            que_type for que_type in examples["question_type"]
        ]
        return examples

    ds = ds.map(ques_type_gene, batched=True, num_proc=12)
    if filter_ques_type:
        ds = ds.filter(
            lambda x: [i == filter_ques_type for i in x["question_type"]],
            batched=True,
        )

    return ds


def load_vizwiz_ds(
    root_dir,
    split=None,
):
    root_dir = Path(root_dir)
    train_ann = root_dir / "train.json"
    val_ann = root_dir / "val.json"

    def preprocess(ann_file):
        return json.load(open(ann_file))

    if split == "train":
        train_ds_list = preprocess(train_ann)
        ds = datasets.Dataset.from_list(train_ds_list)
    elif split == "validation":
        val_ds_list = preprocess(val_ann)
        ds = datasets.Dataset.from_list(val_ds_list)
    else:
        train_ds_list = preprocess(train_ann)
        val_ds_list = preprocess(val_ann)
        ds = datasets.DatasetDict(
            {
                "train": datasets.Dataset.from_list(train_ds_list),
                "validation": datasets.Dataset.from_list(val_ds_list),
            }
        )

    def train_trans(x, idx):
        img_path = [str(root_dir / "train" / f_n) for f_n in x["image"]]

        x["image"] = img_path
        x["idx"] = idx
        x["answer"] = [a[0]["answer"] for a in x["answers"]]
        return x

    def val_trans(x, idx):
        img_path = [str(root_dir / "val" / f_n) for f_n in x["image"]]
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


def load_coco_ds(
    train_coco_dataset_root,
    train_coco_annotation_file,
    val_coco_dataset_root,
    val_coco_annotation_file,
    split=None,
):
    from .coco_dataset import CocoDataset

    if split is None:
        train_ds = CocoDataset(
            train_coco_dataset_root,
            train_coco_annotation_file,
        )
        val_ds = CocoDataset(val_coco_dataset_root, val_coco_annotation_file)
        train_ds = datasets.Dataset.from_list(train_ds)
        val_ds = datasets.Dataset.from_list(val_ds)
        ds = DatasetDict({"train": train_ds, "validation": val_ds})
        ds = ds.sort("image_id")
        ds = ds.cast_column("image", datasets.Image(decode=True))
    else:
        if split == "train":
            ds = CocoDataset(
                train_coco_dataset_root,
                train_coco_annotation_file,
            )
        elif split == "validation":
            ds = CocoDataset(
                val_coco_dataset_root,
                val_coco_annotation_file,
            )
        ds = datasets.Dataset.from_list(ds)
        ds = ds.sort("image_id")
        ds = ds.cast_column("image", datasets.Image(decode=True))
    return ds
