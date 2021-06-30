export CUDA_VISIBLE_DEVICES=$1
worldsize=$2
config=$3
stage='train'
ckpt=$4

python run_dist.py --config $config --stage $stage --world_size $worldsize --ckpt $ckpt