cat=$1
split=$2
gpu=$3
dataset=$4

python extract_mcr_features.py --mode caffe \
         --num-cpus 4 --gpus $gpu \
         --extract-mode roi_feats \
         --min-max-boxes '10,100' \
         --config-file configs/bua-caffe/extract-bua-caffe-r101.yaml \
         --image-dir "./dataset/$dataset/$cat/images/pictures/$split/" \
         --bbox-dir "" \
         --out-dir  "./dataset/$dataset/$cat/images/features/$split/"