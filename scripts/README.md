# Datasets Details

## Requirements

**ICU Tokenizer**

ICU tokenizer supports tokenziation of indonesian and is adopted by [fasttext](https://fasttext.cc).

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

Emoji to Lang is a tool which converts the emoji in different language contexts to according language representations.

```bash
# clone project
git clone https://github.com/jhliu17/emoji-to-lang.git
cd emoji-to-lang

# install package
python setup.py install
```

**Faster RCNN (bottom-up-attention)**

To extract RoI features from the image, we need to adopt a faster rcnn backbone from [bottom-up-attention.pytorch](https://github.com/MILVLG/bottom-up-attention.pytorch). Please follow their installation document to setup Detectron2, Apex, and Ray.


## Download Datasets

Make a data folder `dataset` under your project path.

```bash
mkdir dataset
```

then download the Lazada and Amazon datasets (containing train, dev, and test splits) from [Google Drive](https://drive.google.com/file/d/1XMaospdOeEoXKVuH05YyH038log0lgVx/view?usp=sharing) to `dataset`.

```bash
cd dataset
unzip MRHPDatasets.zip
```

## Datasets Preprocessing

### Step 1: Crawl the product and review image data

Set up the dataset path and related global category settings (i.e. `cat`, `dataset_name`) in `crawl_image.py` and then run the following script to crawl image data. By dafault, it starts a multi-threading (max_workers = 100) program to request desired data.

```bash
python scripts/crawl_data/crawl_image.py
```

After crawling, the image resources are saved in `download_dir` path.

### Step 2: Extract the image features

Copy the feature extraction utils in `scripts/feature_data` dir to the **Faster RCNN (bottom-up-attention)** project folder.

```bash
cp scripts/feature_data/* [path_of_bottom-up-attention.pytorch]
cd [path_of_bottom-up-attention.pytorch]
```

After setting the gpu env and dataset name, run the extraction script.

```bash
sh run_feature_extraction.sh [dataset] [gpu]
```

Dataset path and output path can be modified in `excecute_extraction.sh`.
