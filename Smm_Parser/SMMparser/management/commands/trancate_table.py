from django.core.management.base import BaseCommand, CommandError
from SMMparser.models import Pages, Images
import requests
from SMMparser import utils
from SMMparser.utils import base
from urllib.parse import urlparse, urljoin
import traceback
from django.db import models


class Command(BaseCommand):
    db: models.Model

    """  python manage.py trancate_table Images
    Очищаем таблицы, что у нас есть. указать или Pages или Images или all
    """
    help = "очищаем таблицу"

    def add_arguments(self, parser):
        parser.add_argument("table", type=str, help="называние таблицы")

    def handle(self, *args, **options):
        obj = options['table']

        if obj == 'all':
            Pages.objects.all().delete()
            Images.objects.all().delete()
            return

        db = globals()[obj]
        table = db.objects.all()
        table.delete()
