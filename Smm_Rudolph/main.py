from fastapi import FastAPI
from apps.neural_model_inference.main import router as inference
from apps.neural_model_learn.main import router as learn
import os
app = FastAPI()

app.include_router(inference)
app.include_router(learn)
