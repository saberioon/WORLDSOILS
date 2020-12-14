from sklearn.metrics import mean_squared_error, mean_absolute_error, explained_variance_score
import numpy as np

def MSE(obs, pred):
    """


    :return:
    """
    return mean_absolute_error(obs, pred)

def RMSE(obs, pred):
    """


    :return:
    """
    return np.sqrt(mean_squared_error(obs, pred))

def R2(obs, pred):
    """

    :return:
    """
    return explained_variance_score(obs, pred)