# NTU-CVPDL-HW3
Poject report - https://hackmd.io/@GTooth/Symrqvkrn
## Reminders
* Please start each model reproducement in outest dir.
* Recommend using Conda to manage 2 different envs for each models.
* If you want to leave the current env, use `conda deactivate`.
* I trained models on `CUDA 11.4`, `11G Mem` GPU. You might need to change CUDA version when installing pytorch.
* **For reporducing training please refer to `train.md`.**

## Source Model - Yolov7
### Environment
```bash
conda create -n yolo python=3.10
# press yes for installation
conda activate yolo
pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116
pip install -r yolov7/requirements.txt 
```
### Train
Please refer to `train.md`.

## Domain Adaption Model - MIC
### Environment
```bash
conda create -n mic python=3.9
# press yes for installation
conda activate mic
pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116
bash run_install_for_MIC.sh
```

### Checkpoints
Ckpts will be in `MIC/custom_cache/ckpts`.
```bash
bash hw3_download.sh
```

### Inference
`$CHECKPOINT_INDEX` could be:
* 0 - 0%
* 1 - 33%
* 2 - 66%
* 3 - 100%
* 4 - Best model
```bash
bash hw3_inference.sh $TEST_DIR $JSON_OUTPUT_PATH $CHECKPOINT_INDEX
```

### Train
Please refer to `train.md`.
