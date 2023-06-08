import logging
import os

import numpy as np
import pandas as pd

from skl2onnx import to_onnx
from onnx.onnx_pb import ModelProto

from utils.logging import get_logger
from utils.data import preprop_data
from models.train_sklearn_models import (
    learn_linear_model,
    learn_oversampling_model,
    learn_envelope,
    learn_random_forest
)

logger = get_logger(name=__name__, log_level=logging.DEBUG)


def save_model(model: ModelProto, path:  str):
    with open(path, 'wb') as f:
        f.write(model.SerializeToString())


if __name__ == '__main__':
    df = pd.read_csv('data/creditcard.csv')

    data_train, data_test, target_train, target_test = preprop_data(df)

    # Here we can use different models for deploying
    lg_linear = learn_linear_model(data_train, target_train)
    lg_oversampling = learn_oversampling_model(data_train, target_train)
    ran_for = learn_random_forest(data_train, target_train)

    onx_lg_linear = to_onnx(
        lg_linear, data_train[:1].astype(np.float32), target_opset=12
    )
    onx_lg_oversampling = to_onnx(
        lg_oversampling, data_train[:1].astype(np.float32), target_opset=12
    )
    onx_ran_for = to_onnx(
        ran_for, data_train[:1].astype(np.float32), target_opset=12
    )

    model_dir_path = 'data/models'
    save_model(onx_lg_linear, os.path.join(model_dir_path, 'linear.onnx'))
    save_model(onx_lg_oversampling, os.path.join(model_dir_path, 'oversampling.onnx'))
    save_model(onx_ran_for, os.path.join(model_dir_path, 'random_forest.onnx'))
