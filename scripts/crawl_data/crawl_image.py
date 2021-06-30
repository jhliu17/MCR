from download import (
    ThreadPoolDownloader
)
from download_utils import (
    get_from_logger,
    get_rvw_prd_url,
    pic_statics
)

# global setting
cat = 'home'
dataset_name = 'lazada'


if __name__ == '__main__':
    splits = ['rvw', 'prd']  # ('rvw', 'prd')
    stages = ['train', 'dev', 'test']  # ('train', 'dev', 'test')
    max_workers = 100
    dest_dir = f"./dataset/{dataset_name}/{cat}/images"
    download_dir = f'{dest_dir}/pictures/%s/%s/'

    for stage in stages:
        for split in splits:
            prd_path = f'{dest_dir}/{cat}.prd.%s' % stage
            rvw_path = f'{dest_dir}/{cat}.rvw.%s' % stage
            logger = 'logger.txt'

            dir_path = download_dir % (stage, split)
            split_dl = ThreadPoolDownloader(
                max_workers,
                dir_path,
                logger
            )

            # from dataframe
            if split == 'prd':
                split_to_do = get_rvw_prd_url(prd_path, 'product', True)
            else:
                split_to_do = get_rvw_prd_url(rvw_path, 'review')
            total_pic_num = len(split_to_do)
            split_dl.download(split_to_do)

            # from logger
            # download unsuccessfully image data
            split_to_do = get_from_logger(split_dl.log_file)
            split_dl.download(split_to_do)

            # stat pics number
            down_pic_num = pic_statics(dir_path)
            print(f"Crawl image finished!\nTotal image number {total_pic_num}, successfully download image number {down_pic_num}.")
