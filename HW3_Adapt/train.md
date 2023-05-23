# NTU-CVPDL-HW3
## Reminders
* Please start each model reproducement in outest dir.
* Recommend using Conda to manage 2 different envs for each models.
* If you want to leave the current env, use `conda deactivate`.
* I trained models on `CUDA 11.4`, `11G Mem` GPU. You might need to change CUDA version when installing pytorch.

---

## Train Source Model - Yolov7
### 1. Preprocess
Assume that `hw3_dataset` is unzipped.
```bash
mv hw3_dataset images # rename to meet yolo requirement
python make_yolo_labels.py
```

### 2. Train
The results (including checkpoints every 2 epoch) will lay in `yolov7/runs/yolov7_hw3_source`. Checkpoints(`.pt`) are in `yolov7/runs/yolov7_hw3_source/weights`.
```bash
cd yolov7
bash run_train.sh
cd .. # back to outest dir
```

---

## Train Adaption Model - MIC (Lazy)
```bash
bash hw3_train.sh $TRAIN_DIR $VAL_DIR $OUTPUT_BEST_MODEL_PATH
```

## Train Adaption Model - MIC
### 1. Preprocess
Please make sure there is at least `hw3_dataset` or `images` in your outest dir.
```bash
cd MIC
bash run_train_preprocess.sh
```

### 2. Train
The results (including checkpoints every 1000 iterations) will lay in `MIC/ckpts/mask_06`. Checkpoints(`*.pth`) are also in `MIC/ckpts/mask_06`.
If the batch size is too large for GPU mem, please decrease the hyperparameter in `run_train.sh`
```bash
bash run_train.sh
cd .. # back to outest dir
```