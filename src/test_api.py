import requests

import numpy as np
import pandas as pd

from sklearn.metrics import f1_score

import tkinter as tk
from tkinter import filedialog

from utils.data import preprop_data
from utils.logging import get_logger

logger = get_logger(name=__name__)

if __name__ == '__main__':
    tk.Tk().withdraw()

    logger.info('Choosing data file')
    dataframe = pd.read_csv(filedialog.askopenfilename(
        filetypes=[('CSV DataFrame', '.csv')],
        initialdir='../data/'
    ))

    data_train, data_test, target_train, target_test = preprop_data(dataframe)
    
    predicts = requests.get(
        'http://localhost:8000/prediction/',
        json={'data': data_test.tolist()}
    ).json()

    predicts = np.array(predicts['prediction'])

    logger.info(f'f1 score - {f1_score(target_test, predicts)}')
