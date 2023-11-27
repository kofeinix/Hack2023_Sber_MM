from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import pandas as pd
import random
import PIL
import os
import numpy as np
import pytorch_lightning as pl

class FoodDataset(Dataset):
    def __init__(self, args, file_path, csv_path, tokenizer, shuffle=True):
        self.tokenizer = tokenizer
        self.samples = []
        self.args = args
        self.image_transform = T.Compose([
            T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            T.RandomResizedCrop(self.args.image_size, scale=(1., 1.), ratio=(1., 1.)),
            T.ToTensor()
        ])
        df = pd.read_csv(csv_path)
        df.columns = ['index', 'belok', 'fats', 'uglevod', 'kkal', 'name', 'path']
        for belok, fats, uglevod, kkal, caption, f_path in zip(
            df['belok'],df['fats'], df['uglevod'], df['kkal'], df['name'], df['path']
        ):
            caption = f'блюдо: {caption}; белков: {belok}; жиров: {fats}; углеводов: {uglevod}; ккал: {kkal};'
            if len(caption)>10 and len(caption)<100 and os.path.isfile(f'{file_path}/{f_path}'):
                self.samples.append([file_path, f_path, caption.lower()])
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


class FoodDataModule(pl.LightningDataModule):
  def __init__(self, args, file_path, csv_path, tokenizer):
    super().__init__()
    self.args = args
    self.file_path = file_path
    self.csv_path = csv_path
    self.tokenizer = tokenizer
  def setup(self, stage=None):
    self.train_dataset = FoodDataset(args = self.args,
                                     file_path=self.file_path,
                                     csv_path =self.csv_path,
                                     tokenizer=self.tokenizer)
  def train_dataloader(self):
    return DataLoader(
      self.train_dataset,
      batch_size=self.args.bs,
      shuffle=True,
      drop_last=True
    )
