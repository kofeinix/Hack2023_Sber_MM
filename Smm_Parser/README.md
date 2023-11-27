

sudo curl -sS -o - https://dl-ssl.google.com/linux/linux_signing_key.pub | apt-key add -
sudo echo "deb [arch=amd64]  http://dl.google.com/linux/chrome/deb/ stable main" >> /etc/apt/sources.list.d/google-chrome.list
sudo apt-get -y update
sudo apt-get -y install google-chrome-stable
or
curl -fSsL https://dl.google.com/linux/linux_signing_key.pub | sudo gpg --dearmor | sudo tee /usr/share/keyrings/google-chrome.gpg >> /dev/null
echo deb [arch=amd64 signed-by=/usr/share/keyrings/google-chrome.gpg] http://dl.google.com/linux/chrome/deb/ stable main | sudo tee /etc/apt/sources.list.d/google-chrome.list
sudo apt update
sudo apt-get -y install google-chrome-stable

/usr/bin/google-chrome


https://chromedriver.storage.googleapis.com/
wget https://chromedriver.storage.googleapis.com/114.0.5735.16/chromedriver_linux64.zip
unzip chromedriver_linux64.zip
sudo mv chromedriver /usr/bin/chromedriver
sudo chown root:root /usr/bin/chromedriver
sudo chmod +x /usr/bin/chromedriver

# рабочий для хрома
sudo apt-get -y update
sudo apt-get install chromium-driver
sudo apt-get install xvfb


python 3.9
```
pip install -r requirements.txt
```

**Запустить сервер**

```
python manage.py runserver
```

**Запустить парсер**

Команда принимает URL страницы с товарами и далее будет итерировать по пагинации

`python manage.py parsing https://megamarket.ru/catalog/krossovki-muzhskie -driver t `


**Очистить таблицы с данными**

`python manage.py trancate_table Images` # по конкретной

`python manage.py trancate_table all` # все


**Прочитать картинку из байткода**

```
import io
from PIL import Image
from SMMparser.models import Images
ims = Images.objects.get(id='4262c302-9697-40e2-bf35-c65e2e1fb025')
buf = io.BytesIO(ims.imageio)
image = Image.open(buf)

```
И далее выполняются все манипуляции с экземпляром класса  Image
