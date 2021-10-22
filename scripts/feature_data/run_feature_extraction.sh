dataset=$1
gpu=$2

for cat in clothing electronics home; do
    for split in train dev test; do
        sh excecute_extraction.sh $cat $split $gpu $dataset
    done
done