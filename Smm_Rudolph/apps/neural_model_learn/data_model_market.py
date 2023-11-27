import json

from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import pandas as pd
import random
import PIL
import os
import numpy as np
import pytorch_lightning as pl
from settings.database import session, Pages, Images

class MarketDataset(Dataset):
    def __init__(self, args, file_path, tokenizer, shuffle=True, **kwargs):
        self.tokenizer = tokenizer
        self.samples = []
        self.args = args
        self.image_transform = T.Compose([
            T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            T.RandomResizedCrop(self.args.image_size, scale=(1., 1.), ratio=(1., 1.)),
            T.ToTensor()
        ])

        query = session.query(Images.id, Pages.categ3, Pages.properties, Pages.name, Pages.jsfiled).filter(Images.page_id==Pages.id)
        df = pd.read_sql(query.statement, session.bind)

        for id, categ3, properties, name, jsfiled in zip(
            df['id'], df['categ3'], df['properties'], df['name'], df['jsfiled']
        ):
            jsfiled = json.loads(jsfiled)
            jsfiled_parsed = []
            for key in ['Модель', 'Артикул производителя', 'Бренд', 'Код товара','Страна-производитель',
                         'Тип', 'Пол', 'Сезон', 'Материал верха', 'Цвет']:
                try:
                    jsfiled_parsed.append(f'{key}: {str(jsfiled[key])}')
                except:
                    pass
            jsfiled_parsed= ', '.join(jsfiled_parsed)

            caption = f'имя товара: {name}; {jsfiled_parsed};'
            if  os.path.isfile(f'{file_path}{id}.jpg'):
                self.samples.append([file_path, f'{id}.jpg', caption.lower()])
        if shuffle:
            np.random.shuffle(self.samples)
            print('Shuffled')
    def __len__(self):
        return len(self.samples)
    def load_image(self, file_path, img_name):
        return PIL.Image.open(f'{file_path}/{img_name}')
    def __getitem__(self, item):
        item = item % len(self.samples)
        file_path, img_name, text = self.samples[item]
        try:
            image = self.load_image(file_path, img_name)
            image = self.image_transform(image)
        except Exception as err:
            print(err)
            random_item = random.randint(0, len(self.samples) - 1)
            return self.__getitem__(random_item)
        text = text.lower().strip()
        encoded = self.tokenizer.encode_text(text, text_seq_length=self.args.r_text_seq_length)
        return encoded, image


class MarketDataModule(pl.LightningDataModule):
  def __init__(self, args, file_path, tokenizer, **kwargs):
    super().__init__()
    self.args = args
    self.file_path = file_path
    self.tokenizer = tokenizer
  def setup(self, stage=None):
    self.train_dataset = MarketDataset(args = self.args,
                                     file_path=self.file_path,
                                     tokenizer=self.tokenizer)
  def train_dataloader(self):
    return DataLoader(
      self.train_dataset,
      batch_size=self.args.bs,
      shuffle=True,
      drop_last=True
    )
