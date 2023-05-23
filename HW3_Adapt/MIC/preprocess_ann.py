import json
import os

def modify_ann(ann):
    for i, img in enumerate(ann["images"]):
        img_name = img["file_name"]
        ann["images"][i]["file_name"] = img_name.split("/")[-1]
    for i, _ in enumerate(ann["annotations"]):
        ann["annotations"][i]['segmentation'] = []
    return ann

def main():
    val_ann = json.load(open("./datasets/fog/val.coco.json", 'r'))
    del val_ann["annotations"]; val_ann["annotations"] = []
    ann_count = 0
    del val_ann["images"]; val_ann["images"] = []
    dir_path = "./datasets/fog/train"
    for i, img_name in enumerate(os.listdir(dir_path)):
        val_ann["images"].append({"id": i, "width": 2048, "height": 1024,"file_name": f"{img_name}",})
        val_ann["annotations"].append({"id": ann_count, "image_id": i, "category_id": 1, "iscrowd": 0, "area": 697, "bbox": [715, 397, 20, 53], "segmentation": []})
        ann_count += 1

    open("./datasets/fog/train.coco.json", 'w').write(json.dumps(val_ann, indent=4))

    # modify annotations file name to match the image file name
    # org
    train_ann = json.load(open("./datasets/org/train.coco.json", 'r'))
    train_ann = modify_ann(train_ann)
    open("./datasets/org/train.coco.json", 'w').write(json.dumps(train_ann, indent=4))
    val_ann = json.load(open("./datasets/org/val.coco.json", 'r'))
    val_ann = modify_ann(val_ann)
    open("./datasets/org/val.coco.json", 'w').write(json.dumps(val_ann, indent=4))
    # fog
    #train_ann = json.load(open("./datasets/fog/train.coco.json", 'r'))
    #train_ann = modify_ann(train_ann)
    #open("./datasets/fog/train.coco.json", 'w').write(json.dumps(train_ann))
    val_ann = json.load(open("./datasets/fog/val.coco.json", 'r'))
    val_ann = modify_ann(val_ann)
    open("./datasets/fog/val.coco.json", 'w').write(json.dumps(val_ann, indent=4))

if __name__ == "__main__":
    main()