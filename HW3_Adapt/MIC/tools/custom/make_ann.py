import argparse
import json
import os
import glob

cur_path = os.path.dirname(os.path.abspath(__file__))

def parse_args():
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Training")
    parser.add_argument(
        "--img_dir",
        default=None,
        help="path to img_dir",
    )
    return parser.parse_args()

def main():
    args = parse_args()

    val_ann = json.load(open(os.path.join(cur_path, "val.coco.json"), 'r'))
    del val_ann["annotations"]; val_ann["annotations"] = []
    ann_count = 0
    del val_ann["images"]; val_ann["images"] = []
    dir_path = args.img_dir
    imgs = list(map(lambda x: x.replace(dir_path+'/', ""), glob.glob(f'{dir_path}/**/*.png', recursive=True)))
    for i, img_name in enumerate(imgs):
        val_ann["images"].append({"id": i, "width": 2048, "height": 1024,"file_name": f"{img_name}",})
        val_ann["annotations"].append({"id": ann_count, "image_id": i, "category_id": 1, "iscrowd": 0, "area": 697, "bbox": [715, 397, 20, 53], "segmentation": []})
        ann_count += 1

    output_path = os.path.join(cur_path, "../../custom_cache/custom_ann.coco.json")
    open(output_path, 'w').write(json.dumps(val_ann, indent=4))

if __name__ == "__main__":
    main()