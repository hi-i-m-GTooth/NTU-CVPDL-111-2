import json
import os
import argparse

cur_path = os.path.dirname(os.path.abspath(__file__))

def parse_args():
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Training")
    parser.add_argument(
        "--output",
        default="unnamed.json",
        help="path to output json",
    )
    return parser.parse_args()

def main():
    args = parse_args()

    result = json.load(open(os.path.join(cur_path, "../../custom_cache/inference/custom_test_cocostyle/bbox.json"), 'r'))
    ann = json.load(open(os.path.join(cur_path, "../../custom_cache/custom_ann.coco.json"), 'r'))
    output = dict([(a['file_name'], {'boxes':[], 'labels':[], 'scores':[]}) for a in ann["images"]])

    for r in result:
        image_name = ann['images'][r['image_id']]['file_name']
        yxyx_box = [r['bbox'][0], r['bbox'][1], r['bbox'][0]+r['bbox'][2], r['bbox'][1]+r['bbox'][3]]
        output[image_name]['boxes'].append(yxyx_box)
        output[image_name]['labels'].append(r['category_id'])
        output[image_name]['scores'].append(r['score'])

    open(args.output, 'w').write(json.dumps(output, indent=4))
    print("Successfully write json to {}".format(args.output))

if __name__ == "__main__":
    main()