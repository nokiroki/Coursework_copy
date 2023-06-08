import requests

import numpy as np
import pandas as pd

from sklearn.metrics import f1_score

import tkinter as tk
from tkinter import filedialog

from utils.data import preprop_data
from utils.logging import get_logger
from best_thresholds import *

logger = get_logger(name=__name__)

if __name__ == '__main__':
    tk.Tk().withdraw()

    logger.info('Choosing data file')
    dataframe = pd.read_csv(filedialog.askopenfilename(
        filetypes=[('CSV DataFrame', '.csv')],
        initialdir='../data/'
    ))

    data_train, data_test, target_train, target_test = preprop_data(dataframe)
    
    logger.info('Test prediction methods')

    predicts = requests.get(
        'http://localhost:8000/prediction/',
        params={'name': 'oversampling'},
        json={'data': data_test.tolist(), 'threshold': 0.9999999973801859}
    ).json()

    predicts = np.array(predicts['prediction'])

    logger.info(f'f1 score - {f1_score(target_test, predicts)}')

    logger.info('Test benchmark method')

    info = requests.get(
        'http://localhost:8000/benchmark/',
        params={'name': 'oversampling'},
        json={'data': data_test.tolist(), 'true_targets':target_test.tolist()}
    ).json()

    logger.info(info)
