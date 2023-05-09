import logging

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

import numpy as np
import pandas as pd

from onnxruntime import InferenceSession
from skl2onnx import to_onnx

from utils.logging import get_logger
from utils.data import preprop_data
from models.linear_model import learn_linear_model

logger = get_logger(name=__name__, log_level=logging.DEBUG)


if __name__ == '__main__':
    df = pd.read_csv('../data/creditcard.csv')

    data_train, data_test, target_train, target_test = preprop_data(df)

    # Here we can use different models for deploying
    lg = learn_linear_model(data_train, target_train)

    onx = to_onnx(lg, data_train[:1].astype(np.float32), target_opset=12)

    model_path = '../data/logreg.onnx'
    with open(model_path, 'wb') as f:
        f.write(onx.SerializeToString())

    sess = InferenceSession(model_path)
    pred_onnx = sess.run(None, {'X': data_test.astype(np.float32)})[0]

    pred_sklearn = lg.predict(data_test)

    logger.info(
        f'F1 difference of predictions - {f1_score(pred_onnx, pred_sklearn)}'
    )
