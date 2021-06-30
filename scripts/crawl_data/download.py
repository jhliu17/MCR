import os
import time
import typing
import random

import requests
from tqdm import tqdm
from concurrent import futures
from download_utils import process_url
from fake_useragent import UserAgent


class BaseDownloader(object):
    def __init__(self, dest_dir, log_file, timeout=30):
        if not os.path.isdir(dest_dir):
            os.makedirs(dest_dir)

        self.dest_dir = dest_dir
        self.log_file = os.path.join(dest_dir, log_file)
        self.time_out = timeout
        self._write_logger('')
        self._write_logger(time.strftime("%Y-%m-%d, %H:%M:%S"))
    
    def _write_logger(self, *strings):
        string = ','.join(strings)
        
        with open(self.log_file, 'a+') as f:
            f.write(string + '\n')

    def _end_download(self):
        self.close_logger()

    def download(self, download_list):
        download_list = self._remove_exist(download_list)
        
        t0 = time.time()
        count = len(download_list)
        print("Try to download %d object" % count)
        if count > 0:
            success_count = self.run(download_list)
        else:
            success_count = 0
        elapsed = time.time() - t0
        msg = '\n%d/%d imgs downloaded in %.2f senconds' % (success_count, count, elapsed)
        print(msg)
        
        self._write_logger(msg)
    
    def _remove_exist(self, download_list):
        candidate = []

        for url, filename in download_list:
            path = os.path.join(self.dest_dir, filename)
            if not os.path.isfile(path):
                candidate.append((url, filename))
        return candidate
        
    def run(self, wait_list) -> int:
        """ return success instance"""
        raise NotImplementedError
    
    def save_img(self, img, filepath):
        filedir, filename = os.path.split(filepath)

        # make item dir
        path = os.path.join(self.dest_dir, filedir)
        try:
            if not os.path.exists(path):
                os.mkdir(path)
        except FileExistsError:
            pass
        
        # get file path
        path = os.path.join(path, filename)
        state = True
        try:
            if not os.path.isfile(path):
                with open(path, 'wb') as fp:
                    fp.write(img)
        except:
            # print('Failed to save', path)
            state = False
        return state

    def get_img(self, url, headers=None) -> typing.Tuple[bool, typing.Any]:
        try:
            if headers:
                resp = requests.get(url, headers=headers, timeout=self.time_out)
            else:
                resp = requests.get(url, timeout=self.time_out)

            if resp.status_code == 200:
                return True, resp.content
            else:
                raise ValueError
        except:
            # print('Failed to access', url)
            return False, None


class SequenceDownloader(BaseDownloader):
    def run(self, wait_list):
        success = 0
        for url, filename in wait_list:
            state, img = self.get_img(url)
            
            if not state:
                self._write_logger(url)
                continue
            
            self.save_img(img, filename)
            success += 1
        return success


class ThreadPoolDownloader(BaseDownloader):
    def __init__(self, max_worker, dest_dir,
                 log_file, random_sleep=False, fake_head=False):
        super().__init__(dest_dir, log_file)
        self.max_worker = max_worker
        self.random_sleep = random_sleep
        self.ua = UserAgent() if fake_head else None

    def _get_fake_header(self):
        headers = None
        if self.ua:
            headers = {'User-Agent': self.ua.random}
        return headers
    
    def _random_sleep(self):
        if self.random_sleep:
            sec = random.uniform(0, 0.5)
            time.sleep(sec)

    def _run_one(self, args):
        url, filename = args
        url = process_url(url)

        self._random_sleep()
        headers = self._get_fake_header()
        
        # get image
        state, img = self.get_img(url, headers=headers)
        if not state:
            print('Failed to access', url)
            self._write_logger('Failed access', url, filename)
            return 0
        
        # save image
        state = self.save_img(img, filename)
        if not state:
            print('Failed to save', filename)
            self._write_logger('Failed saving', url, filename)
            return 0

        return 1
    
    def run(self, wait_list):
        workers = min(self.max_worker, len(wait_list))

        with futures.ThreadPoolExecutor(workers) as executor:
            res = list(tqdm(executor.map(self._run_one, wait_list), total=len(wait_list)))
        
        return sum(res)


class VerboseThreadPoolDownloader(ThreadPoolDownloader):
    def run(self, wait_list):
        workers = min(self.max_worker, len(wait_list))

        success_num = 0
        with futures.ThreadPoolExecutor(workers) as executor:
            to_do_map = {}
            for wl in wait_list:
                future = executor.submit(self._run_one, wl)
                to_do_map[future] = wl

            done_iter = futures.as_completed(to_do_map)
            done_iter = tqdm(done_iter, total=len(wait_list))
            
            for future in done_iter:
                res = future.result()
                success_num += res
            
        return success_num
