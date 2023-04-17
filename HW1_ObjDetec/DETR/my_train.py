from transformers import AutoModelForObjectDetection, TrainingArguments, Trainer, EvalPrediction
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torch.utils.data import DataLoader
import torch
from pathlib import Path
from datasets import Dataset

from my_preprocess import *
from my_dataset import MarinDataset



train_path = Path("../images/train/_annotations.coco.json")
valid_path = Path("../images/valid/_annotations.coco.json")
train_set, valid_set = MarinDataset(train_path), MarinDataset(valid_path)
id2label, label2id = train_set.make_labelmaps()
train_set, valid_set = Dataset.from_list([i for i in train_set.tolist() if len(i["objects"]["bbox"])]), Dataset.from_list(valid_set.tolist())
train_set, valid_set = train_set.with_transform(transform_aug_ann), valid_set.with_transform(transform_aug_ann)


model = AutoModelForObjectDetection.from_pretrained(
    ckpt,
    id2label=id2label,
    label2id=label2id,
    ignore_mismatched_sizes=True,
).to(device="cuda")


training_args = TrainingArguments(
    output_dir="./runs/detr-resnet-101_finetuned_marin_lr00004",
    per_device_train_batch_size=4,
    num_train_epochs=100,
    evaluation_strategy="epoch",
    #eval_steps = 1,
    logging_strategy="epoch",
    save_strategy="epoch",
    fp16=True,
    learning_rate=1e-4,
    weight_decay=1e-3,
    #save_total_limit=2,
    remove_unused_columns=False,
    push_to_hub=False,
    #metric_for_best_model='mAP@[50:5:95]',
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=collate_fn,
    train_dataset=train_set,
    eval_dataset=valid_set,
    tokenizer=image_processor,
    #compute_metrics=compute_metrics,
)

trainer.train()


