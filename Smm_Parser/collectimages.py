from test import d_init
from SMMparser.utils import ParsItem, Pages, Images
from PIL import Image
import io
from io import BytesIO
import requests

images = Images.objects.filter(imageio__isnull=True)

for img in images:
    content = requests.get(img.url, verify=False).content
    im = Image.open(BytesIO(content))
    im = im.resize((190, 190))
    buf = io.BytesIO()
    im.save(buf, format='JPEG')  # subsampling=0, quality=100
    # img.imageio = buf.getvalue()
    # img.save()
print()
