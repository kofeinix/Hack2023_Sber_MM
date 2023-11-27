# from SMMparser import utils
# from SMMparser.utils import base, ParsList
from urllib.parse import urlparse, urljoin
from fake_useragent import UserAgent
# from SMMparser.utils import parse_by_pagenation_page, parse_items_list, get_content, get_driver
from selenium.webdriver.chrome.options import Options
from selenium import webdriver
from xvfbwrapper import Xvfb
from save_screen import full_screenshot_with_scroll
from selenium.webdriver.common.by import By
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.support import expected_conditions as ec


ua = UserAgent()
#vdisplay = Xvfb(width=1280, height=720)
vdisplay = Xvfb(width=1920, height=1080)
vdisplay.start()


#url = "https://amiunique.org/fingerprint"
url = "https://megamarket.ru/catalog/krossovki-muzhskie"
#url = "https://megamarket.ru"

_url = urlparse(url)
root_url = f"{_url.scheme}://{_url.netloc}"


options = Options()
options.add_argument(f'user-agent={ua.chrome}')
options.add_argument('--disable-dev-shm-usage')
# options.add_argument('DNT=1')
#options.add_experimental_option("prefs", {"enable_do_not_track": True})
# if headless:
#     options.add_argument('--headless')
#     options.add_argument('--no-sandbox')
options.page_load_strategy = 'normal'  # 'normal'
driver = webdriver.Chrome(options=options)
driver.set_window_size(1920, 1080)
driver.implicitly_wait(60)
print()
driver.get(root_url)
driver.save_screenshot('screen.png')

driver.get(url)
driver.save_screenshot('screen2.png')
elem = driver.find_element(By.CLASS_NAME, 'sticky-element-wrapper')

# try:
#     element = WebDriverWait(driver, 30).until(ec.presence_of_element_located((By.XPATH, '//*[@id="app"]/main/div/div[1]/div[2]/div/div[2]/div[2]/div[2]/div[3]/div/div')))
# finally:
#     driver.quit()


# print(driver.page_source)


el = driver.find_element(By.TAG_NAME, 'body')
el.screenshot('screen_elem.png')

path = Path('full_screen.png')
full_screenshot_with_scroll(driver, path)


# def interceptor(request):
#     del request.headers['Referer']  # Delete the header first
#     request.headers['Referer'] = 'some_referer'

# Set the interceptor on the driver
# driver.request_interceptor = interceptor


print("sd")
driver.quit()
vdisplay.stop()
