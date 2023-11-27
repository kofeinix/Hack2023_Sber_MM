import datetime
import os
import torch
import uvicorn
from rudalle import get_tokenizer, get_vae
import pytorch_lightning as pl
from rudolph.model import get_rudolph_model, ruDolphModel, FP16Module
from settings.learn_settings import *
from .data_models_food import FoodDataset, FoodDataModule
from .data_model_market import MarketDataset, MarketDataModule
from .train_class import Rudolph_
from fastapi import FastAPI, APIRouter
from pytorch_lightning.loggers import WandbLogger
import wandb

router = APIRouter(
    prefix="/learn",
    tags=["learn"],
    responses={404: {"description": "Not found"}},
)

def set_args():
    model = get_rudolph_model(model_name,  fp16=True, device=device)
    tokenizer = get_tokenizer()
    vae = get_vae(dwt=False).to(device)
    args = Args(model)
    if not os.path.exists(args.model_save_path):
        os.makedirs(args.model_save_path)
    return tokenizer, args, vae


def prepare_data(file_paths, dataset, data_module, tokenizer, args, vae):
    dataset = dataset(args=args, file_path=file_paths['file_path'] ,csv_path =file_paths['csv_path'] ,tokenizer=tokenizer)
    args.train_steps = len(dataset)//args.bs
    data_module = data_module(args=args, file_path=file_paths['file_path'] ,csv_path =file_paths['csv_path'],tokenizer=tokenizer)
    Rudolph_.vae = vae
    model = Rudolph_(model_name, device, args)
    return data_module, model

def fit(args, model,data_module, model_file_name):
    try:
        wandb_logger = WandbLogger(project='my-awesome-project')
        wandb_logger.experiment.config["batch_size"] = args.bs
        trainer = pl.Trainer(
          logger=wandb_logger,
          #checkpoint_callback=checkpoint_callback,
          max_epochs=args.epochs,
          accelerator=device,
          #progress_bar_refresh_rate=30
        )
        trainer.fit(model, data_module)
        trainer.save_checkpoint(args.checkpont_save_path+f'lastchkp_{model_file_name}.pt')
        return True
    except Exception as e:
        print(e)
        return False

def _fix_pl(args, path, model_file_name):
  d = torch.load(path)["state_dict"]
  checkpoint = {}
  for key in d.keys():
    checkpoint[key.replace('model.','')] = d[key]
  torch.save(checkpoint, args.model_save_path + f'{model_file_name}')

@router.post("/")
def train(model, model_file_name):
    if model == 'food':
        dataset = FoodDataset
        data_module = FoodDataModule
        file_paths = path['food']
    elif model == 'market':
        dataset = MarketDataset
        data_module = MarketDataModule
        file_paths = path['market']
    else:
        raise
    print('get tokens and args')
    tokenizer, args, vae = set_args()
    print('get models')
    data_module, model = prepare_data(file_paths, dataset, data_module, tokenizer, args, vae)
    print('fitting')
    status = fit(args, model, data_module, model_file_name)

    if status:
        print('fixing data and saving model')
        _fix_pl(args, args.checkpont_save_path+f'lastchkp_{model_file_name}.pt')
    return status

