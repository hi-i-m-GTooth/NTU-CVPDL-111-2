batch_size=4
name=mask_06

python tools/train_net.py --config-file configs/da_faster_rcnn/e2e_da_faster_rcnn_R_50_FPN_masking_cs.yaml \
                            OUTPUT_DIR ckpts/${name} SOLVER.CHECKPOINT_PERIOD 1000 SOLVER.MAX_ITER 60000 SOLVER.IMS_PER_BATCH $batch_size 