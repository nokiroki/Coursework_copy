from fastapi import FastAPI
from pydantic import BaseModel

import numpy as np

from onnxruntime import InferenceSession

import tkinter as tk
from tkinter import filedialog


tk.Tk().withdraw()

sess = InferenceSession(filedialog.askopenfilename(
    filetypes=[('ONNX model', '.onnx')],
    initialdir='../../data/'
))

app = FastAPI()

class InferenceDataFrame(BaseModel):
    data: list[list[float]]

@app.get('/')
async def root():
    return {'greeting': 'Hello world!'}

@app.get('/prediction/')
async def prediction(data_frame: InferenceDataFrame):
    data = np.array(data_frame.data, dtype=np.float32)
    preds: np.ndarray = sess.run(None, {'X': data})[0]
    return {'prediction': preds.tolist()}
