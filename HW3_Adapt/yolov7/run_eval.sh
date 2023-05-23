name=adapt_test_source
model=./runs/train/yolov7_hw3_source/weights/epoch_015.pt
test=../images/org/val

rm -rf ./runs/detect/${name}

python eval.py --weights ${model} --source ${test} --img-size 1024 --device 0 \
		--hyp data/hyp.scratch.custom.yaml --data data/source.yaml --batch-size 1 \
		--name ${name} --conf-thres 0.25 --img-size 2048

name=adapt_target
test=../images/fog/val
rm -rf ./runs/detect/${name}

python eval.py --weights ${model} --source ${test} --img-size 1024 --device 0 \
		--hyp data/hyp.scratch.custom.yaml --data data/target.yaml --batch-size 1 \
		--name ${name} --conf-thres 0.25 --img-size 2048
