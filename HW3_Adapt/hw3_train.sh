train_dir=$1
train_dir=$(realpath $train_dir)
val_dir=$2
val_dir=$(realpath $val_dir)
best_model=$3
best_model=$(realpath $best_model)

# hyperparameters
batch_size=4
name=mask_06

cd MIC
cp -r $train_dir .
mv ${train_dir##*/} hw3_dataset
bash run_train_preprocess.sh
bash run_train.sh
cp ckpts/mask_06/model_final.pth $best_model
cd ..