img_dir=$1
img_dir=$(realpath $img_dir)
output=$2
output=$(realpath $output)
ckpt=$3

# hyperparameters
batch_size=1

cd MIC
# preprcess img_dir
python tools/custom/make_ann.py --img_dir $img_dir

# inference
python tools/test_net.py --config-file "configs/da_faster_rcnn/e2e_da_faster_rcnn_R_50_FPN_masking_cs.yaml" \
                        MODEL.WEIGHT custom_cache/ckpts/${ckpt}.pth TEST.IMS_PER_BATCH $batch_size \
                        OUTPUT_DIR custom_cache TEST.CUSTOM_DIR $img_dir
# edit output json
python tools/custom/result2json.py --output $output

cd ..