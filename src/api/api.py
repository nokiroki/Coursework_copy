import os
from typing import Optional

from fastapi import FastAPI
from pydantic import BaseModel

import numpy as np

from sklearn.metrics import (
    precision_recall_curve,
    f1_score,
    roc_auc_score,
    precision_score,
    recall_score
)

from onnxruntime import InferenceSession

import tkinter as tk
from tkinter import filedialog


tk.Tk().withdraw()

app = FastAPI()

class InferenceDataFrame(BaseModel):
    data: list[list[float]]
    true_targets: Optional[list[float]] = None
    threshold: Optional[float] = None

def get_probs(probs_list: list[dict[int, float]]) -> np.ndarray:
    probs = np.zeros(len(probs_list))

    for i, class_probs in enumerate(probs_list):
        probs[i] = class_probs[1]

    return probs

def calculate_with_tr(probs: np.ndarray, tr: float) -> np.ndarray:
    return (probs >= tr).astype(np.int32)

def get_optimum_threshold(y_true: np.ndarray, y_pred_proba: np.ndarray) -> float:
    precisions, recalls, trs = precision_recall_curve(
        y_true, y_pred_proba
    )

    f1_scores = 2 * (precisions * recalls) / (precisions + recalls)

    return trs[np.argmax(f1_scores)]

def get_metrics(test_true: np.ndarray, test_scores: np.ndarray)-> tuple[
    float, float, float, float, float
]:
    trs = get_optimum_threshold(test_true, test_scores)
    # print(f'Optimal threshold is - {trs}\n')
    
    test_pred = test_scores >= trs

    return (
        trs,
        f1_score(test_true, test_pred),
        roc_auc_score(test_true, test_scores),
        precision_score(test_true, test_pred),
        recall_score(test_true, test_pred)
    )

@app.get('/')
async def root():
    return {'greeting': 'Hello world!'}

@app.get('/prediction/')
async def prediction(data_frame: InferenceDataFrame, name: str = 'linear'):
    data = np.array(data_frame.data, dtype=np.float32)
    sess = InferenceSession(os.path.join(f'../../data/models/{name}.onnx'))
    if data_frame.threshold:
        probs = get_probs(sess.run(None, {'X': data})[1])
        preds = calculate_with_tr(probs, data_frame.threshold)
    else:
        preds: np.ndarray = sess.run(None, {'X': data})[0]
    return {'prediction': preds.tolist()}

@app.get('/benchmark/')
async def benchmark(data_frame: InferenceDataFrame, name: str = 'linear'):
    data = np.array(data_frame.data, dtype=np.float32)
    true_targets = np.array(data_frame.true_targets, dtype=np.float32)

    sess = InferenceSession(os.path.join(f'../../data/models/{name}.onnx'))
    probs = get_probs(sess.run(None, {'X': data})[1])
    metrics = get_metrics(true_targets, probs)

    json = {
        'threshold': metrics[0],
        'f1': metrics[1],
        'rocauc': metrics[2],
        'precision': metrics[3],
        'recall': metrics[4]
    }

    return json
