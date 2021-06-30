# env setting
export NLTK_DATA='/home/junhao.jh/bin/nltk'


export CUDA_VISIBLE_DEVICES=$1
config=$2
stage='train'

python run.py --config $config --stage $stage