import argparse
import os
import json
import cv2

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='exp', help='project/name')
    parser.add_argument('--source', type=str, default='../images/test', help='source')
    parser.add_argument('--json_name', default='exp', help='json name')

    args = parser.parse_args()

    print("Convert to JSON.")
    rlt = {f: {"boxes": [], "labels":[], "scores":[]} for f in os.listdir(args.source) if ".jpg" in f}

    full_pred_dir = os.path.join("runs/detect/", args.name, "labels")
    for fn in os.listdir(full_pred_dir):
        f = open(os.path.join(full_pred_dir, fn), 'r')
        ls = f.readlines()
        jpg_fn = fn.replace(".txt", ".jpg")
        jpg_path = os.path.join(args.source, jpg_fn)
        h, w, _ = 1, 1, 1#cv2.imread(jpg_path).shape
        for l in ls:
            ts = l.strip().split(' ')
            rlt[jpg_fn]["boxes"].append([float(ts[1])*w, float(ts[2])*h, float(ts[3])*w, float(ts[4])*h])
            rlt[jpg_fn]["labels"].append(int(ts[0]))
            rlt[jpg_fn]["scores"].append(float(ts[5]))
    
    json.dump(rlt, open(args.json_name, 'w'))

