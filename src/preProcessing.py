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


def scaling_y_data(data_frame: pd.DataFrame) -> pd.DataFrame:
    """

    :param data_frame:
    :return:
    """

    robustScaler_y = robust_scale(data_frame)

    return robustScaler_y
