if [[ -d "../images" ]]; then
    echo "../images exists."
    cp -r ../images/org/ ./datasets/org/
    cp -r ../images/fog/ ./datasets/fog/
fi

if [[ -d "../hw3_dataset" ]]; then
    echo "../hw3_dataset exists."
    cp -r ../hw3_dataset/org/*.json ./datasets/org/
    cp -r ../hw3_dataset/fog/*.json ./datasets/fog/
fi

python3 preprocess_ann.py