import yaml
import os
import json
from pathlib import Path

yml_path = "./data/marin.yaml"

if __name__ == "__main__":
    yml = yaml.safe_load(open(yml_path, 'r'))
    for dataset in ["train", "val"]:
        print(f"Make labels of {dataset} ...")
        dir_path = Path(yml[dataset])
        store_dir_path = Path(yml[dataset].replace("images", "labels"))
        store_dir_path.mkdir(parents=True, exist_ok=True)
        
        anno_path = Path(os.path.join(dir_path, "_annotations.coco.json"))
        anno = json.load(open(anno_path, 'r'))
        img_labels = [[] for i in range(len(anno['images']))]

        for a in anno["annotations"]:
            img_labels[a["image_id"]].append({"category_id": a['category_id'], "bbox": a["bbox"]})

        for id in range(len(img_labels)):
            img_name = anno["images"][id]["file_name"]
            file_path = Path(os.path.join(store_dir_path, img_name.replace(".jpg", ".txt")))
            h, w = anno["images"][id]['height'], anno["images"][id]['width']
            f = open(file_path, 'w')
            if not img_labels[id]:
                f.write(f"0 0 0 0 0\n")
                continue
            for l in img_labels[id]:
                bbox = l['bbox']
                bbox[0], bbox[1], bbox[2], bbox[3] = (bbox[0]+bbox[2]/2)/w, (bbox[1]+bbox[3]/2)/h, bbox[2]/w, bbox[3]/h
                f.write(f"{l['category_id']} {' '.join(map(lambda x: str(x), bbox))}\n")
            f.close()

        


