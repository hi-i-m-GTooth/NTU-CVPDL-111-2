train=1024
test=1024
#python train.py --epochs 100 --workers 4 --device 0 --batch-size 64 \
#--data data/marin.yaml --img 640 640 --cfg cfg/training/yolov7-tiny-marin.yaml \
#--weights 'yolov7-tiny.pt' --name yolov7_tiny_marin_res --hyp data/hyp.scratch.tiny.yaml

python train.py --epochs 200 --workers 16 --device 0 --batch-size 4 \
--data data/marin.yaml --img ${train} ${test} --cfg cfg/training/yolov7-marin.yaml \
--weights 'yolov7_training.pt' --name yolov7_marin_res --hyp data/hyp.scratch.custom.yaml #\
#--multi-scale