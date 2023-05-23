name=adapt_test_target
model=../best_model.pt
test=$1
#json_name=$2

rm -rf ./runs/detect/${name}

python detect.py --weights ${model} --source ${test} --img-size 1024 --device 0 \
		--name ${name} --conf-thres 0.25 --img-size 2048 #--save-txt --save-conf --no-trace --exist-ok

#python detect2json.py --name ${name} --source ${test} --json_name ${json_name}
