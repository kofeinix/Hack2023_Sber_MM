import d_init
import requests
from SMMparser import utils
from SMMparser.utils import base, ParsList
from urllib.parse import urlparse, urljoin
import traceback
from SMMparser.models import Images
import io
from PIL import Image
from io import BytesIO
import os

# ims = Images.objects.all()[0]
# Кроссовки New Balance Fresh Foam X More V4, синий
# 4262c302-9697-40e2-bf35-c65e2e1fb025

# ims = Images.objects.get(id='4262c302-9697-40e2-bf35-c65e2e1fb025')
# response = requests.get(ims.url)
# # im = Image.open('test.jpg')
# im = Image.open(BytesIO(response.content))
# im = im.resize((190, 190))
# buf = io.BytesIO()
# im.save(buf, format='JPEG')  # subsampling=0, quality=100
# ims.imageio = buf.getvalue()
# ims.save()
#
#
# buf = io.BytesIO(ims.imageio)
# image = Image.open(buf)

# from SMMparser.models import Pages
# page = Pages.objects.get(name="Кроссовки New Balance Fresh Foam X More V4, синий")

# URL = "https://megamarket.ru/catalog/?q=%D0%BA%D1%80%D0%BE%D1%81%D1%81%D0%BE%D0%B2%D0%BA%D0%B8%20fila%20chain%20mid%20wntr,%20%D1%87%D0%B5%D1%80%D0%BD%D1%8B%D0%B9"
# URL = "https://megamarket.ru/catalog/?q=%D0%BA%D1%80%D0%BE%D1%81%D1%81%D0%BE%D0%B2%D0%BA%D0%B8%20pulse%2018-12mv-005st,%20%D1%85%D0%B0%D0%BA%D0%B8"

from SMMparser.utils import parse_by_pagenation_page, parse_items_list, get_content, get_driver
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from fake_useragent import UserAgent

url = "https://megamarket.ru/catalog/krossovki-muzhskie"
url_ = urlparse(url)
utils.URL = f"{url_.scheme}://{url_.netloc}"
utils.HEADLESS = True
utils.ISDRIVER = True

ua = UserAgent()
content = get_content(url)
soup = BeautifulSoup(content, "html.parser")
items = soup.find("div", {"class": "catalog-listing__items catalog-listing__items_divider"})
items = items.find_all("div", {"class": "catalog-item catalog-item-desktop ddl_product catalog-item_enlarged"})

# browser = get_driver()


for item in items:
    try:
        # ParsList(item, browser).run()
        object = ParsList(item)
        object.run() if object else None
    except Exception as e:
        print(traceback.format_exc())
        continue
print()
browser.quit()
