#!/usr/bin/env python
# coding:utf-8

import os
import logging

logging_level = {'debug': logging.DEBUG,
                 'info': logging.INFO,
                 'warning': logging.WARNING,
                 'error': logging.ERROR,
                 'critical': logging.CRITICAL}


def debug(msg):
    logging.debug(msg)
    print('DEBUG: ', msg)


def info(msg):
    logging.info(msg)
    print('INFO: ', msg)


def info_format(msg):
    logging.info(msg)
    print('INFO FORMAT:')
    print(msg)


def warning(msg):
    logging.warning(msg)
    print('WARNING: ', msg)


def error(msg):
    logging.error(msg)
    print('ERROR: ', msg)


def fatal(msg):
    logging.critical(msg)
    print('FATAL: ', msg)


class Logger(object):
    def __init__(self, config):
        """
        set the logging module
        :param config: helper.configure, Configure object
        """
        super(Logger, self).__init__()
        assert config.log.level in logging_level.keys()
        logging.getLogger('').handlers = []
        
        log_dir, _ = os.path.split(config.log.filename)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)

        logging.basicConfig(filename=config.log.filename,
                            level=logging_level[config.log.level],
                            format='%(asctime)s - %(levelname)s : %(message)s',
                            datefmt='%Y/%m/%d %H:%M:%S')
