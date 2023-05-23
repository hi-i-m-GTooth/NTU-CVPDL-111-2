name=60k
python tools/test_net.py --config-file "configs/da_faster_rcnn/e2e_da_faster_rcnn_R_50_FPN_masking_cs.yaml" \
                        MODEL.WEIGHT ckpts/${name}/model_0055000.pth TEST.IMS_PER_BATCH 1 \
                        OUTPUT_DIR cutom_cache DATASETS.CUSTOM_DIR 
