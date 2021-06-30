import os
import json
import pandas as pd

from tqdm import tqdm


def process_url(url: str):
    # when w/o http
    if not url.startswith('http'):
        url = url.lstrip('/')
        url = 'http://' + url
    return url


def parse_url(string) -> list:
    try:
        url = json.loads(string)
    except json.JSONDecodeError:
        return []
    
    if isinstance(url, str):
        url = [url]
    return url


def get_rvw_prd_url(datapath, key, flatten=False):
    frame = pd.read_json(datapath)

    todo = []
    for _, line in tqdm(frame.iterrows(), total=len(frame)):
        item_id = line['%s_id' % key]
        img_url_string = line['img_url']
        img_url_list = parse_url(img_url_string)

        total_img = len(img_url_list)
        for i, img_url in enumerate(img_url_list):
            if flatten:
                url = img_url
            else:
                url = img_url['url']
            filepath = "%s/pic_%d_%d.jpg" % (item_id, total_img, i)
            todo.append((url, filepath))
    
    return todo


def get_from_logger(logger):
    todo = []
    with open(logger, 'r') as f:
        added = set()
        for line in tqdm(f):
            line = line.strip()
            if line.startswith('Failed'):
                line = line.split(',')
                url = line[1]
                filepath = line[2]

                if filepath not in added:
                    added.add(filepath)
                    # process url
                    url = process_url(url)
                    todo.append((url, filepath))

    return todo


def pic_statics(download_path):
    down_pic = 0
    for root, dirs, files in os.walk(download_path):
        for f in files:
            if f.endswith('.jpg'):
                down_pic += 1
    return down_pic


def clean_invalid_pic(download_path):
    clean_pic = 0
    for root, dirs, files in os.walk(download_path):
        for f in files:
            if f.endswith('\n'):
                full_path = os.path.join(root, f)
                os.remove(full_path)
                clean_pic += 1
    return clean_pic


def parse_amazon_url(string) -> list:    
    url = string
    if isinstance(url, str):
        url = [url]

    # replace finger photo into big image file
    big_url = []
    for u in url:
        if u:
            try:
                *left, mid, right = u.split(".")
                left = ".".join(left)
                big_url.append(f"{left}.{right}")
            except:
                print(u)
                continue
    return big_url


def get_amazon_rvw_prd_url(datapath, key, flatten=True):
    frame = pd.read_json(datapath)

    todo = []
    for _, line in tqdm(frame.iterrows(), total=len(frame)):
        item_id = line['%s_id' % key]
        img_url_string = line['img_url']
        img_url_list = parse_amazon_url(img_url_string)

        total_img = len(img_url_list)
        for i, img_url in enumerate(img_url_list):
            if flatten:
                url = img_url
            else:
                url = img_url['url']
            filepath = "%s/pic_%d_%d.jpg" % (item_id, total_img, i)
            todo.append((url, filepath))
    
    return todo