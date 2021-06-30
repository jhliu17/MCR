# Datasets

More details comming soon


## Requirements

**ICU Tokenizer**

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

**Emoji**

```bash
git clone https://github.com/jhliu17/emoji-to-lang.git
cd emoji-to-lang
python setup.py install
```

**Faster RCNN (bottom-up-attention)**

To extract RoI features from the image, we need to adopt a faster rcnn backbone from [bottom-up-attention.pytorch](). Use the following scripts to execute the extraction program.

```bash
git clone https://github.com/jhliu17/emoji-to-lang.git
cd emoji-to-lang
python setup.py install
```


## Download Datasets

You can download the Lazada and Amazon datasets (containing train, dev, and test splits) from [Google Drive]().


## Processing Steps

Step 1: Crawl the product and review image data.

```
python scripts/crawl_data/crawl_image.py
```

Step 2: Extract the review image features.

```
python scripts/feature_data/extract_feature.py
```
