# Datasets Details

## Requirements

**ICU Tokenizer**

ICU tokenizer supports tokenization of Indonesian. This tokenizer is adopted by [fasttext](https://fasttext.cc) for processing multi-lingual corpus.

```bash
# conda install icu libarary
conda install icu pkg-config

# Or if you wish to use the latest version of the ICU library, the conda-forge channel typically contains a more up to date version.
conda install -c conda-forge icu

# mac os
CFLAGS="-std=c++11" PATH="/usr/local/opt/icu4c/bin:$PATH" \
    pip install ICU-Tokenizer

# ubuntu
CFLAGS="-std=c++11" pip install ICU-Tokenizer
```

**Emoji to Lang**

Emoji to Lang is a tool converting the emoji in different language contexts into according language representations.

```bash
# clone project
git clone https://github.com/jhliu17/emoji-to-lang.git
cd emoji-to-lang

# install package
python setup.py install
```

**Faster RCNN (bottom-up-attention)**

To extract RoI features from the image, we need to adopt a faster rcnn backbone from [bottom-up-attention.pytorch](https://github.com/MILVLG/bottom-up-attention.pytorch). Please follow their [installation document](https://github.com/MILVLG/bottom-up-attention.pytorch#prerequisites) to setup **Detectron2**, **Apex**, and **Ray**.


## Download Datasets

Make a data folder `dataset` under your project path,

```bash
mkdir dataset
```

then download the Lazada and Amazon datasets (containing train, dev, and test splits) from [Google Drive](https://drive.google.com/file/d/1XMaospdOeEoXKVuH05YyH038log0lgVx/view?usp=sharing) to the `dataset` dir.

```bash
cd dataset
unzip MRHPDatasets.zip
```

## Datasets Preprocessing

### Step 1: Crawl product and review image data

Set up the dataset path and related global category settings (i.e. `cat`, `dataset_name`) in `crawl_image.py` and then run the following script to crawl image data. By dafault, it starts a multi-threading (max_workers = 100) program to request desired data.

```bash
python scripts/crawl_data/crawl_image.py
```

The image resources are saved in `download_dir` path.

### Step 2: Extract image features

Copy the feature extraction utils in `scripts/feature_data` dir to the **Faster RCNN (bottom-up-attention)** project folder.

```bash
cp scripts/feature_data/* [path_of_bottom-up-attention.pytorch]
cd [path_of_bottom-up-attention.pytorch]
```

After setting the gpu env and dataset name, run the extraction script to generate RoI features. Dataset path and output path can be modified in `excecute_extraction.sh`.

```bash
sh run_feature_extraction.sh [dataset] [gpu]
```

### Step 3: Unify image package with npz

To pack the image features under a review or product, we provide a pack script to unify them.

```bash
python scripts/utils/unify_features.py
```


## Citations

If you use our datasets, please cite these papers using BibTeX references below.

```bibtex
@inproceedings{mcr,
    title={Multi-perspective Coherent Reasoning for Helpfulness Prediction of Multimodal Reviews},
    author={Junhao Liu, Zhen Hai, Min Yang, and Lidong Bing},
    booktitle={Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics, {ACL} 2021},
    year={2021},
}

@inproceedings{amazon18,
    title={Justifying recommendations using distantly-labeled reviews and fined-grained aspects},
    author={Jianmo Ni, Jiacheng Li, and Julian McAuley},
    booktitle={Proceedings of Empirical Methods in Natural Language Processing, {EMNLP} 2019},
    year={2019},
}
```