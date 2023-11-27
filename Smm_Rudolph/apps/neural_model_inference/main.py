import io

import requests
import uvicorn
from PIL import Image
import torch
from rudalle import get_tokenizer, get_vae
from rudalle.tokenizer import YTTMTokenizerWrapper
from rudolph.model import get_rudolph_model
from rudolph.api import ruDolphApi

from apps.analyzer.analyzer import fresh_train, use_predictor
from settings.config import KEYS, EXCEL_LEARN, DATA_PATH
from settings.learn_settings import *

from fastapi import FastAPI, APIRouter, UploadFile
run_app = FastAPI()
model=None

model_data = fresh_train(keys=KEYS, excel=EXCEL_LEARN, path=DATA_PATH)

router = APIRouter(
    prefix="/inference",
    tags=["inference"],
    responses={404: {"description": "Not found"}},
)

def init_model(model_file_name):
    model = get_rudolph_model(model_name,  fp16=True, device=device)
    args = Args(model)
    name = model_file_name#args.model_name
    model_path = args.model_save_path + name + '.pt'
    tokenizer = get_tokenizer()
    vae = get_vae(dwt=False).to(device)
    model.load_state_dict(torch.load(model_path))
    return model, tokenizer, vae

loaded_models = {}
@router.get("/by_url")
def predict(model_file_name='awesomemodel', captions_num=4, img_by_url = 'https://img.delo-vcusa.ru/2020/11/Borshh-s-yablokami.jpg',  template = 'категория'):
    predict = use_predictor(model_data=model_data, keys=KEYS, excel=True, path=DATA_PATH)
    predictor = predict['smm']
    if model_file_name in loaded_models:
        model, tokenizer, vae = loaded_models[model_file_name]
    else:
        try:
            model, tokenizer, vae = init_model(model_file_name)
            loaded_models[model_file_name]=(model, tokenizer, vae)

        except:
            print('no model found!')
            raise
    img_by_url = Image.open(requests.get(img_by_url, stream=True).raw).resize((128, 128))
    api = ruDolphApi(model, tokenizer, vae, bs=48)
    texts, counts = api.generate_captions(img_by_url, r_template=template,
                          top_k=16, captions_num=int(captions_num), bs=16, top_p=0.6, seed=43,
                          temperature=0.8)
    ppl_txt, ppl_img, scores = api.self_reranking_by_image(texts, img_by_url, bs=16, seed=42)
    answer = []
    for idx in ppl_img.argsort():
        print(texts[idx])
        data = text_to_analyzer(predictor, texts[idx])
        links = [x['url'] for x in data]
        res={}
        res['text'] = texts[idx]
        res['links'] = links
        answer.append(res)
    return answer

@router.post("/by_image")
async def predict(img: UploadFile, model_file_name='awesomemodel', captions_num=4,  template = 'категория'):
    predict = use_predictor(model_data=model_data, keys=KEYS, excel=True, path=DATA_PATH)
    predictor = predict['smm']
    if model_file_name in loaded_models:
        model, tokenizer, vae = loaded_models[model_file_name]
    else:
        try:
            model, tokenizer, vae = init_model(model_file_name)
            loaded_models[model_file_name]=(model, tokenizer, vae)
        except:
            print('no model found!')
            raise
    img_object_content = await img.read()
    img = Image.open(io.BytesIO(img_object_content)).resize((128, 128))
    api = ruDolphApi(model, tokenizer, vae, bs=48)
    texts, counts = api.generate_captions(img, r_template=template,
                          top_k=16, captions_num=int(captions_num), bs=16, top_p=0.6, seed=43,
                          temperature=0.8)
    ppl_txt, ppl_img, scores = api.self_reranking_by_image(texts, img, bs=16, seed=42)
    answer = []
    for idx in ppl_img.argsort():
        print(texts[idx])
        data = text_to_analyzer(predictor, texts[idx])
        links = [x['url'] for x in data]
        res={}
        res['text'] = texts[idx]
        res['links'] = links
        answer.append(res)
    return answer

def text_to_analyzer(predictor, text):
    print('predictor run')
    return predictor.predict(description=text, from_file=True)
