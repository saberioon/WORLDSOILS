# SPDX-FileCopyrightText:`2020 Helmholtz Centre Potsdam - GFZ German Research Centre for Geosciences Potsdam, Germany'
# SPDX-License-Identifier: MIT

# !/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
This script analysis LUCAS-15/bssl dataset based on Fully connected neural network

written by : Mohammadmehdi Saberioon
revised date: 01.01.2021

"""


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


def RPD(obs, pred):
    """


    :return:
    """
    mse = mean_squared_error(obs, pred)
    rpd = obs.std() / np.sqrt(mse)
    return rpd
