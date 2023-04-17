name=marin_infer_valid_174
model=../best_model.pt
test=$1
json_name=$2

rm -rf ./runs/detect/${name}

python detect.py --weights ./runs/train/yolov7_marin_res/weights/best.pt --source ${test} --img-size 1024 --device 0 \
		--name ${name} --conf-thres 0.15 --img-size 1024 --save-txt --save-conf --no-trace --exist-ok

python detect2json.py --name ${name} --source ${test} --json_name ${json_name}
