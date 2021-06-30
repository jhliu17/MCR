export CUDA_VISIBLE_DEVICES=$1
config=$2
stage='test'
ckpt=$3

python run.py --config $config --stage $stage --ckpt $ckpt