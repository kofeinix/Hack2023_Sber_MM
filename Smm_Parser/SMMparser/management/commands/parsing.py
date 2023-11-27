from django.core.management.base import BaseCommand, CommandError
# from SMMparser.models import Pages, Images
import requests
from SMMparser import utils
from SMMparser.utils import base
from urllib.parse import urlparse, urljoin
import traceback
import argparse


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


class Command(BaseCommand):
    """  python .\manage.py parsing https://megamarket.ru/catalog/krossovki-muzhskie """
    help = "Парсим магазин"

    def add_arguments(self, parser):
        parser.add_argument("url", type=str, help="укажите url страницы. и парсинг будет идти по пагинации")
        parser.add_argument("-driver", dest="driver", type=str2bool, help="использовать селениум ?", default=False)
        parser.add_argument("-headless", dest="headless", type=str2bool, help="headless mode", default=False)

    def handle(self, *args, **options):
        url = options['url']
        url_ = urlparse(url)
        utils.URL = f"{url_.scheme}://{url_.netloc}"
        utils.URL = f"{url_.scheme}://{url_.netloc}"
        utils.ISDRIVER = options['driver']
        utils.HEADLESS = options['headless']

        try:
            base(url)
        except Exception as e:
            self.stdout.write(traceback.format_exc())
            # self.stdout.write("успешно")
