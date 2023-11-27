from test import d_init
from SMMparser.utils import ParsItem, Pages, get_driver

item_name = 'Кроссовки Ralf Ringer 138105ЧН, черный'
page, created = Pages.objects.get_or_create(name=item_name)
print()

# driver = get_driver(type="firefox")
driver = get_driver()
pars = ParsItem(page.url, driver, page)
pars.run()
