# SPDX-FileCopyrightText:`2020 Helmholtz Centre Potsdam - GFZ German Research Centre for Geosciences Potsdam, Germany'
# SPDX-License-Identifier: MIT

# !/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
This script analysis LUCAS-15/bssl dataset based on Fully connected neural network

written by : Mohammadmehdi Saberioon
revised date: 01.01.2021

"""

import pandas as pd
from sklearn.preprocessing import robust_scale
from sklearn.preprocessing import StandardScaler


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


