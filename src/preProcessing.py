# SPDX-FileCopyrightText:`2020 Helmholtz Centre Potsdam - GFZ German Research Centre for Geosciences Potsdam, Germany'
# SPDX-License-Identifier: EUPL-1.2

# !/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
This script analysis LUCAS-15/bssl dataset based on Fully connected neural network

written by : Mohammadmehdi Saberioon
revised date: 01.01.2021

"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import robust_scale, StandardScaler, MinMaxScaler


def scaling_y_data(data_frame: pd.DataFrame) -> pd.DataFrame:
    """
    Scalling dependent variable

    :param data_frame:
    :return:
    """

    robustScaler_y = robust_scale(data_frame)

    return robustScaler_y


def scaler_data(data_frame: pd.DataFrame) -> pd.DataFrame:
    # reshape 1d arrays to 2d arrays
    data_frame = data_frame.reshape(len(data_frame), 1)

    scaler = StandardScaler()
    scaler.fit(data_frame)
    data_frame_scaled = scaler.transform(data_frame)

    return data_frame_scaled


def scaler_min_max_x_data(data_frame: pd.DataFrame) -> pd.DataFrame:

    scaler = MinMaxScaler()
    data_frame = scaler.fit_transform(data_frame)

    return data_frame


def scaler_min_max_y_data(data_frame: pd.DataFrame) -> pd.DataFrame:

    scaler = MinMaxScaler()
    data_frame = np.reshape(data_frame, (-1, 1))
    # data_frame = data_frame.reshape(-1, 1)
    data_frame = scaler.fit_transform(data_frame)

    return data_frame
