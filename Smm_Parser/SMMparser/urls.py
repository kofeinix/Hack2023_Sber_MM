import io
from PIL import Image
from io import BytesIO

from django.http import HttpResponse
from django.template.defaulttags import url
from .views import get_image



# url(r'^get_image', get_image)
