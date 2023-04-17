from transformers import AutoModelForObjectDetection, TrainingArguments, Trainer, EvalPrediction
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torch.utils.data import DataLoader
from torchvision.ops import nms
import torch
from pathlib import Path
from datasets import Dataset
from PIL import ImageDraw
from tqdm import tqdm

from my_preprocess import *
from my_dataset import MarinEvalDataset, MarinDataset


val_path = Path("../images/valid/")
val_set = MarinEvalDataset(val_path)

model = AutoModelForObjectDetection.from_pretrained(
    "/home/guest/r11922a16/git/NTU-CVPDL-111-2/HW1_ObjDetec/DETR/runs/detr-resnet-101_finetuned_marin_lr00004/checkpoint-5600",
    #id2label=id2label,
    #label2id=label2id,
    #ignore_mismatched_sizes=True,
).to(device="cuda")

mAP = MeanAveragePrecision(box_format='xyxy', iou_type='bbox').to(device="cuda")
model.eval()
with torch.no_grad():
    for i in tqdm(range(len(val_set))):
        #print(batch) # pixel_values, pixel_masks, labels 
        #print(labels)
        d = val_set[i]
        inputs = image_processor(images=d["image"], return_tensors="pt").to("cuda")
        outputs = model(**inputs)
        #outputs["logits"] = outputs["logits"].to("cpu")
        target_sizes = torch.Tensor([[d['height'], d['width']]])            
        results = image_processor.post_process_object_detection(outputs, threshold=0.2, target_sizes=target_sizes)[0]

        nms_indexes = nms(scores = results["scores"], boxes = results["boxes"], iou_threshold=0.3)
        results["scores"] = torch.Tensor([s.item() for i, s in enumerate(results["scores"]) if i in nms_indexes])
        results["boxes"] = torch.Tensor([s.tolist() for i, s in enumerate(results["boxes"]) if i in nms_indexes])
        results["labels"] = torch.Tensor([s.item() for i, s in enumerate(results["labels"]) if i in nms_indexes])
        d["target"]["boxes"], d["target"]["labels"] = torch.Tensor(d["target"]["boxes"]), torch.Tensor(d["target"]["labels"])
        mAP.update([results], [d["target"]])

print("\n\nMetircs:", mAP.compute())