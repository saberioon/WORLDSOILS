import pandas as pd
import numpy as np

np.random.seed([3, 14])
left = pd.DataFrame(data={'value': np.random.randn(4)}, index=['A', 'B', 'C', 'D'])
right = pd.DataFrame(data={'value': np.random.randn(4)},  index=['B', 'D', 'E', 'F'])
left.index.name = right.index.name = 'idxkey'

# SPDX-FileCopyrightText:
# SPDX-License-Identifier: MIT

# !/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
This script analysis LUCAS-15 dataset based on Fully connected neural network

written by : Mohammadmehdi Saberioon
revised date: 08.12.2020

"""

import pandas as pd


def prepare_data_set(data_frame: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare the raw data by :
        - splitting to calibration and test

    Arg:
        data_frame: A panda DataFrame.

    Returns:
        Two panda DataFrames 1.Calibration 2.Test
    """
    data_frame = data_frame.set_index("ID")

    data_frame = data_frame.merge(data_frame, left_index=True, right_index=True)

    return data_frame


if __name__ == "__main__":
    prepare_data_set()



import pandas as pd

df = pd.read_csv("~/OneDrive - Jihočeská univerzita v Českých Budějovicích/Projects/WORLDSOILS/data/clhs_lucas15.csv").set_index("ID")

df2 = pd.read_csv("~/OneDrive - Jihočeská univerzita v Českých Budějovicích/Projects/WORLDSOILS/data/S2a_resampled_lucas15.csv").set_index("PointID")

df.index.name = df2.index.name = 'ID'



df3 = df.merge(df2, left_index=True, right_index=True)

df3.set_index("split").loc["calibration"].describe()


df_cali = df3.set_index("split").loc["calibration"]
df_test = df3.set_index("split").loc["test"]
