from transformers import AutoModelForObjectDetection, TrainingArguments, Trainer, EvalPrediction
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torch.utils.data import DataLoader
from torchvision.ops import nms
import torch
from pathlib import Path
from datasets import Dataset
from PIL import ImageDraw
import argparse, os, json

from my_preprocess import *
from my_dataset import MarinInferDataset, MarinDataset


parser = argparse.ArgumentParser()
parser.add_argument('--test', default='exp', help='test images dir')
parser.add_argument('--json_name', default='pred.json', help='json name')
args = parser.parse_args()

test_path = Path(args.test)
test_set = MarinInferDataset(test_path)

model = AutoModelForObjectDetection.from_pretrained(
    "./runs/detr-resnet-101_finetuned_marin_lr00004/checkpoint-5600",
    #id2label=id2label,
    #label2id=label2id,
    #ignore_mismatched_sizes=True,
).to(device="cuda")

#print(valid_set[0])
rlt = {f: {"boxes": [], "labels":[], "scores":[]} for f in os.listdir(args.test) if ".jpg" in f}
model.eval()
with torch.no_grad():
    for i in range(len(test_set)):
        #print(batch) # pixel_values, pixel_masks, labels 
        #print(labels)
        d = test_set[i]
        inputs = image_processor(images=d["image"], return_tensors="pt").to("cuda")
        outputs = model(**inputs)
        #outputs["logits"] = outputs["logits"].to("cpu")
        target_sizes = torch.Tensor([[d['height'], d['width']]])            
        results = image_processor.post_process_object_detection(outputs, threshold=0.2, target_sizes=target_sizes)[0]

        nms_indexes = nms(scores = results["scores"], boxes = results["boxes"], iou_threshold=0.3)
        image = d['image']
        draw = ImageDraw.Draw(image)
        for index, score, label, box in zip(range(len(results["scores"])), results["scores"], results["labels"], results["boxes"]):
            if index in nms_indexes:
                box = [round(i, 2) for i in box.tolist()]
                x, y, x2, y2 = tuple(box)
                draw.rectangle((x, y, x2, y2), outline="red", width=1)
                draw.text((x, y), model.config.id2label[label.item()], fill="white")
                rlt[d["image_name"]]["boxes"].append([x, y, x2, y2])
                rlt[d["image_name"]]["labels"].append(label.item())
                rlt[d["image_name"]]["scores"].append(score.item())
        image.save("./runs/infer/"+d["image_name"])
        print(d["image_name"])
json.dump(rlt, open(args.json_name, 'w'))

