test_dir=$(readlink -f $1)
output=$(readlink -f $2)

pushd yolov7
./run_detect.sh $test_dir $output
popd