# NTU-CVPDL-111-2-HW1

# Infer images and output results in JSON
In this project, we will infer images via our best model checkpoint, which is trained from **Yolov7**.
## 1. Environment
Python == `3.10`, Troch == `1.13.0`. Note that I use `cu117` for Torch.
```bash
$ pip install -r requirements.txt
```
## 2. Download models
```bash
$ ./hw1_download.sh
```
## 3. Inference
```bash
$ ./hw1.sh $IMG_DIR_PATH $JSON_OUTPUT_PATH
```

---

# Reproduce
## 1. Yolov7 (CNN-based Model)
The training results will be stored at `yolov7/runs/train`.
```bash
$ cd yolov7
$ ./run_train.sh
```
## 2. DETR (Transformer-based Model)
The training results will be stored at `DETR/runs`.
```bash
$ cd DETR
$ ./run_my_train.sh
```
