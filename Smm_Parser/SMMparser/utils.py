from selenium import webdriver
import traceback
import io
from io import BytesIO
from multiprocessing.pool import ThreadPool
from bs4 import BeautifulSoup
import requests
from .models import Pages, Images
from urllib.parse import urljoin, urlparse
import urllib3
from fake_useragent import UserAgent
from PIL import Image
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from pprint import pprint
from datetime import time
import time
import sys
import platform
from selenium.webdriver.common.by import By
from time import sleep

urllib3.disable_warnings()
ua = UserAgent()

URL = None
ISDRIVER = False    # условие использовать селениум или реквест
HEADLESS = False    # открывать окно браузера или нет

categ_dict = {
    0: "categ_root",
    1: "categ1",
    2: "categ2",
    3: "categ3",
    4: "categ4"
}


def get_driver(headless=False, type: str = ''):
    options = Options()
    # options.add_argument(f'user-agent={ua.random}')
    options.add_argument(f'user-agent={ua.chrome}')
    options.add_argument('--disable-dev-shm-usage')
    if headless:
        options.add_argument('--headless')
        options.add_argument('--no-sandbox')
    options.page_load_strategy = 'eager'  #'normal'
    # options.page_load_strategy = 'normal'  #'normal'
    options.add_experimental_option("prefs", {"enable_do_not_track": True})
    if type.lower() == 'firefox':
        driver = webdriver.Firefox(options=options)
    else:
        driver = webdriver.Chrome(options=options)
    # driver = webdriver.Chrome(executable_path="./drivers/chromedriver", options=options)
    # driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    driver.set_window_size(1920, 1080)
    driver.implicitly_wait(10)
    return driver

# driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

def get_page(url, driver):
    driver.get(url)
    file = url.split('/')[-1] + '.png'
    # driver.save_screenshot(file)
    driver.execute_script("window.localStorage.clear();")  # работает
    driver.delete_all_cookies()


def get_content_driver(url, driver):
    get_page(url, driver)
    content = driver.page_source
    return content

# browser.execute_script("return window.localStorage;")
# browser.get_cookies()
# driver.execute_script("window.localStorage.setItem('key','value');")
# driver.execute_script("window.localStorage.getItem('key');")


def get_content(url):
    if ISDRIVER:
        browser = get_driver(HEADLESS)
        return get_content_driver(url, browser)
    else:
        page = requests.get(url, verify=False, headers={'User-Agent': ua.random})
        return page.content


def base(url):
    if platform.system() == "Linux" and not HEADLESS:
        from xvfbwrapper import Xvfb
        vdisplay = Xvfb(width=1920, height=1080)
        vdisplay.start()
    _url = urlparse(url)
    root_url = f"{_url.scheme}://{_url.netloc}"
    driver = get_driver()
    # driver.get(root_url)
    get_page(root_url, driver)
    driver.find_element(By.CLASS_NAME, 'promo-constructor')
    driver.save_screenshot('root_screen.png')
    run(url, driver)


def run(url, driver):
    num, url = parse_by_pagenation_page(url, driver)
    # sys.exit()
    while num and url:
        time.sleep(5)
        url = urljoin(URL, url)
        print(num, url)
        num, url = parse_by_pagenation_page(url, driver)


def get_active_pagin(pagis):
    for i, li in enumerate(pagis):
        active = li.attrs.get('class')
        if active and active[0] == "active":
            return i+1


# def parse_by_pagenation_page(url, driver: False):
#     """ страница с пагинацией """
#     content = get_content_driver(url, driver) if driver else get_content(url)
#     soup = BeautifulSoup(content, "html.parser")
#     parse_items_list(soup)
#     pagis = soup.find('nav', {"class": "pager catalog-listing__pager"}).find_all('li')
#     num = get_active_pagin(pagis)
#     try:
#         return num, pagis[num].a['href']
#     except Exception as e:
#         print("больше нет пагинации")
#         return False, False


def parse_by_pagenation_page(url, driver: False):
    """ страница с пагинацией """
    driver.get(url)
    elem = driver.find_element(By.CLASS_NAME, "catalog-listing__items")
    content = driver.page_source
    soup = BeautifulSoup(content, "html.parser")
    parse_items_list(soup, driver)
    pagis = soup.find('nav', {"class": "pager catalog-listing__pager"}).find_all('li')
    num = get_active_pagin(pagis)
    try:
        return num, pagis[num].a['href']
    except Exception as e:
        print("больше нет пагинации")
        return False, False


def parse_items_list(soup, driver):
    """ страница со списком. парсим объекты """

    # items = soup.find_all("div", {"compare-onboarding-name": "productCompareListingTour"})
    items = soup.find("div", {"class": "catalog-listing__items catalog-listing__items_divider"})
    items = items.find_all("div", {"class": "catalog-item catalog-item-desktop ddl_product catalog-item_enlarged"})

    # ThreadPool(10).map(ParsList, items)
    for item in items:
        try:
            item_name = item.find("div", {"class": "item-title"}).text.strip()
            page_db, created = Pages.objects.get_or_create(name=item_name)
            if not created:
                print(f"Пропускаем. {item_name} есть в БД")
                continue  # если есть в БД, то пропускаем
            url = item.find('a', {'class': "item-image-block ddl_product_link"})
            href = url.get('href')
            url = urljoin(URL, href)
            print(f"Добавлям. {item_name} {url}")
            page_db.url = url
            page_db.save()
            sleep(2)
            object = ParsItem(page_db, driver)
            # object = ParsItem(page_db)
            object.run() if object else None
        except Exception as e:
            print(traceback.format_exc())
            continue


class ParsList:
    page_db: Pages

    def __new__(cls, item, driver=None):
        instance = object.__new__(cls)
        item_name = item.find("div", {"class": "item-title"}).text.strip()
        page_db, created = Pages.objects.get_or_create(name=item_name)
        if not created:
            print(f"Пропускаем. {item_name} есть в БД")
            return  # если есть в БД, то пропускаем
        print(f"Добавляем. {item_name}")
        instance.page_db = page_db
        if driver:
            instance.browser = driver
        else:
            instance.browser = get_driver(HEADLESS)
        return instance

    def __init__(self, item, driver=None):
        self.item = item

    def run(self):
        self.parse_item()
        self.page_db.save()

    def parse_item(self):
        """ переходим на страницу предметно для этого ищем url """
        url = self.item.find('a', {'class': "item-image-block ddl_product_link"})
        href = url.get('href')
        url = urljoin(URL, href)
        self.page_db.url = url
        self.page_db.save()
        print("Добавляем ", self.page_db.name, self.page_db.url, self.page_db.link2())
        self.parse_page(url)

    def parse_page(self, url):
        """ обрабатываем страницу с товаром """
        content = get_content_driver(url, self.browser)
        page_soup = BeautifulSoup(content, "html.parser")
        self.pars_nav(page_soup)
        self.parse_images(page_soup)
        self.pars_properties(page_soup)

    def parse_images(self, page_soup):
        """ только эталонные картинки. исключаем пользовательские. условие по pic """
        object = page_soup.find('div', {"class": "pdp-first-screen-fashion"})
        images = [i for i in object.find_all('img') if i.get('alt') == "pic"]
        for img in images:
            try:
                src = img.get('src')
                image, created = Images.objects.get_or_create(url=src, page_id=self.page_db.id)
                if created:
                    content = requests.get(src, verify=False, headers={'User-Agent': ua.random}).content
                    # content = get_content(src)
                    # im = Image.open('test.jpg')
                    im = Image.open(BytesIO(content))
                    im = im.resize((190, 190))
                    buf = io.BytesIO()
                    im.save(buf, format='JPEG')  # subsampling=0, quality=100
                    image.imageio = buf.getvalue()
                    image.save()

            except Exception as e:
                continue

    def pars_properties(self, page_soup):
        """ дергаем характеристики """
        # pdp-specs__list
        properties = page_soup.find("div", {"class": "pdp-specs__list"})
        text = properties.text.replace("\t", "").replace("\n  \n", ": ")
        self.page_db.properties = text

    def pars_nav(self, page_soup):
        """ парсим навигацию Одежда, обувь и аксессуары, Мужская обувь, Кроссовки мужские
        и т.д.
        """
        nav = page_soup.find_all('li', {"class": "breadcrumb-item"})
        print()
        for i, j in enumerate(nav):
            if not categ_dict.get(i):
                continue
            if not j.a:
                continue
            value = j.a.text
            atr = categ_dict[i]
            setattr(self.page_db, atr, value)

    def complete(self):
        pass


class ParsItem:

    def __init__(self, db_obj, browser=False, ):
        # self.url = url
        self.browser = browser if browser else get_driver()
        self.db_obj = db_obj
        self.url = self.db_obj.url
        print()

    def run(self):
        # self.db_obj.save()
        # self.browser.get(self.url)
        get_page(self.url, self.browser)
        self.pars_nav()
        self.parse_images()
        self.pars_properties()
        self.db_obj.save()

    def pars_nav(self):
        nav = self.browser.find_elements(By.CLASS_NAME, "breadcrumb-item")
        if not nav:
            return

        for i, j in enumerate(nav):
            if not categ_dict.get(i):
                continue
            # a = j.find_element(By.TAG_NAME, 'a')
            # if not a:
            #     continue
            # value = a.text
            value = j.text
            atr = categ_dict[i]
            setattr(self.db_obj, atr, value)

    def parse_images(self):
        images = self.browser.find_element(By.CLASS_NAME, "pdp-video-player-container")
        all = images.find_elements(By.TAG_NAME, 'img')
        for img in all:
            src = img.get_attribute('src')
            alt = img.get_property('alt')
            if not alt:
                continue

            image, created = Images.objects.get_or_create(url=src, page_id=self.db_obj.id)
            # continue
            # if created:
            #     content = requests.get(src, verify=False, headers={'User-Agent': ua.random}).content
            #     # content = get_content(src)
            #     # im = Image.open('test.jpg')
            #     im = Image.open(BytesIO(content))
            #     im = im.resize((190, 190))
            #     buf = io.BytesIO()
            #     im.save(buf, format='JPEG')  # subsampling=0, quality=100
            #     image.imageio = buf.getvalue()
            #     image.save()

    def pars_properties(self):
        properties = self.browser.find_element(By.CLASS_NAME, "pdp-specs__list")
        self.db_obj.properties = properties.text

                # nav = page_soup.find_all('li', {"itemprop": "itemListElement"})
    # nav = object.find_all('ul', {"itemscope": "itemscope"})
    # breadcrumbs
# breadcrumbs__content

# def parse_bar(soup):
#     """ не актуально """
#     bar2 = soup.find('div', {"class": "catalog-collections-selector-item__content"})
#     span = bar2.find_all('span', {"class": "catalog-collections-selector-item__title-text"})
#     for i in span:
#         print(i.text)

# def parse_items(soup):
#     items = soup.find_all("div", {"compare-onboarding-name": "productCompareListingTour"})
#     for item in items:
#         title = item.find("div", {"class": "item-title"}).text.strip()  # название товара
#         div_picts = item.find("div", {"class": "item-image item-image_fashion item-image_changeable"})
#         img_picts = [_ for _ in div_picts.find_all("img",) if _.get('alt')]
#         for pic in img_picts:
#             parse_picture(pic)
#
#
# def parse_picture(pic):
#     photo_url = pic.find("img", )

