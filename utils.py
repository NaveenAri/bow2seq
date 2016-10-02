import logging
import requests
import os

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

class Logger(object):
    DEFAULT_LEVEL = logging.INFO + 1
    logging.addLevelName(DEFAULT_LEVEL, "Bow2Seq")

    @classmethod
    def initialize(cls, filename):
        logging.basicConfig(level=cls.DEFAULT_LEVEL, filename=filename)

    @classmethod
    def log(cls, msg, *args, **kwargs):
        print msg
        logging.log(cls.DEFAULT_LEVEL, '\n'+msg, *args, **kwargs)

def get_from_url(url, cookies=None):
    """
    :param url: url to download from
    :return: return the content at the url
    """
    return requests.get(url, cookies=cookies).content

def get_data_or_download(dir_name, file_name, url='', size='unknown',
                         cookies=None):
    """Returns the data. if the data hasn't been downloaded, then first download the data.
    :param dir_name: directory to look in
    :param file_name: file name to retrieve
    :param url: if the file is not found, then download it from this url
    :param size: the expected size
    :param cookies: Cookies for the request in case of download
    :return: path to the requested file
    """
    dname = os.path.join('data', dir_name)
    fname = os.path.join(dname, file_name)
    if not os.path.isdir(dname):
        assert url, 'Could not locate data {}, and url was not specified. Cannot retrieve data.'.format(dname)
        os.makedirs(dname)
    if not os.path.isfile(fname):
        assert url, 'Could not locate data {}, and url was not specified. Cannot retrieve data.'.format(fname)
        logging.warning('downloading from {}. This file could potentially be *very* large! Actual size ({})'.format(url, size))
        with open(fname, 'wb') as f:
            f.write(get_from_url(url, cookies=cookies))
    return fname