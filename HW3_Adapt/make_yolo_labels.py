import yaml
import os
import json
from pathlib import Path

yml_path = "./yolov7/data/source.yaml"
tar_yml_path = "./yolov7/data/target.yaml"

if __name__ == "__main__":
    src_yml = yaml.safe_load(open(yml_path, 'r'))
    tar_yml = yaml.safe_load(open(tar_yml_path, 'r'))
    for domain, dataset in [("src", "train"), ("src", "val"), ("tar", "val")]:
        print(f"Make labels of {domain}-{dataset} ...")
        if domain == "src":
            yml = src_yml
        else:
            yml = tar_yml
        dir_path = Path(yml[dataset].replace("../", "./"))
        store_dir_path = Path(yml[dataset].replace("../", "./").replace("images", "labels"))
        store_dir_path.mkdir(parents=True, exist_ok=True)
        
        anno_path = Path(str(dir_path)+".coco.json")
        anno = json.load(open(anno_path, 'r'))
        img_labels = [[] for i in range(len(anno['images']))]

        for a in anno["annotations"]:
            img_labels[a["image_id"]].append({"category_id": a['category_id'], "bbox": a["bbox"]})

        for id in range(len(img_labels)):
            img_name = anno["images"][id]["file_name"]
            file_path = Path(os.path.join(store_dir_path, Path(img_name.replace(".png", ".txt")).name))
            h, w = anno["images"][id]['height'], anno["images"][id]['width']
            f = open(file_path, 'w')
            if not img_labels[id]:
                f.write(f"0 0 0 0 0\n")
                continue
            for l in img_labels[id]:
                bbox = l['bbox']
                bbox[0], bbox[1], bbox[2], bbox[3] = (bbox[0]+bbox[2]/2)/w, (bbox[1]+bbox[3]/2)/h, bbox[2]/w, bbox[3]/h
                f.write(f"{int(l['category_id'])-1} {' '.join(map(lambda x: str(x), bbox))}\n")
            f.close()

        


