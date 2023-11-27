Пре-реквизиты:
1. Если рабочая машинка на Windows, то необходима установка Visual Studio 2022. При установке он спросит, какие конкретно модули нужны - нужно выбрать то что связано с разработкой на C/C++
2. На Linux необходимо сделать apt install gcc, apt install g++
3. Необходимо установить CUDA11.8 и удостовериться, что путь к ней есть в переменной Path
4. Установить RUST, убедиться что все галки про добавление в Path стоят при установке
5. pip install --upgrade pip setuptools wheel
6. pip install cython
7. Потом попробовать установить requirements. С этим могут возникнуть определенные проблемы
   а) один пакет может просить при установке версию другого пакета, которая не такая как в requirements
   б) может ругаться на отсутсвие Cython
Самые вредные модули - это transformers и tokenizers  (проблема с версиями). С версиями из requirements все работает, но их может быть сложно установить. Я при наличии проблем скачивал нужную версию пакета в формате zip/tar, заходил внутрь и правил внутренние requirements,
после чего устанавливал из архива. 


Запуск:
в files создать папки checkpoints и models
python -m uvicorn main:app --port xxxx
потом нужно запустить обучение - в браузере http://localhost:xxxx/learn/food (или market) (делал под сваггер, не успел доделать)
предсказания пока тоже не особо оформил, работают для food для одной картинки из интернета, которую я захардкодил
бд в корне можно менять, для дозагрузки изображений нужно использовать apps/images_loader/loader.py 