from django.conf import settings
import os
print(os.getcwd())
os.environ["DJANGO_SETTINGS_MODULE"] = "mysite.settings"
# settings.configure()
from urllib3 import disable_warnings
import django
django.setup()
