from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from bs4 import BeautifulSoup
import requests
from pprint import pprint
# browser = webdriver.Edge()
browser = webdriver.Chrome()
browser.get('https://megamarket.ru/catalog/krossovki-muzhskie')
browser.find_element_


from django.utils.html import html_safe
from django.utils.safestring import mark_safe

URL = "https://megamarket.ru/catalog/krossovki-muzhskie"
page = requests.get(URL, verify=False)

# print(page.text)

soup = BeautifulSoup(page.content, "html.parser")


# mydivs = bs4.find_all("div", {"class": "stylelistrow"})

bar = soup.find_all('nav', {"class": "breadcrumbs catalog-default__breadcrumbs"})
# bar = soup.find('div', {"class": "catalog-department-header catalog-default__department-header container"})
bar2 = soup.find('div', {"class": "catalog-collections-selector-item__content"})
span = bar2.find_all('span', {"class": "catalog-collections-selector-item__title-text"})
for i in span:
    print(i.text)


# all_page_1 = soup.find_all("div", {"class": "compare-onboarding-name"})  # картинки
items = soup.find_all("div", {"compare-onboarding-name": "productCompareListingTour"})  # объект товара
obj1 = items[0]
title = obj1.find("div", {"class": "item-title"}).text.strip()
# link = title.find("a", {''})

mydivs = obj1.find("div", {"class": "item-image item-image_fashion item-image_changeable"})  # картинки

pprint(mydivs.prettify())

# photos = mydivs.find_all("img", {"class": "lazy-img"})
photos2 = [_ for _ in mydivs.find_all("img",) if _.get('alt')]



print('')
