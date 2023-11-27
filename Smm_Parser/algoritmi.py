# from SMMparser.models import Pages, Images
from test import d_init
import requests
from SMMparser import utils
from SMMparser.utils import base
from urllib.parse import urlparse, urljoin
import traceback

url = 'https://megamarket.ru/catalog/krossovki-muzhskie'
url_ = urlparse(url)
utils.URL = f"{url_.scheme}://{url_.netloc}"
utils.ISDRIVER = True
utils.HEADLESS = True

base(url)

