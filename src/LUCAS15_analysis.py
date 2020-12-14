# SPDX-FileCopyrightText:`2020 Helmholtz Centre Potsdam - GFZ German Research Centre for Geosciences Potsdam, Germany'
# SPDX-License-Identifier: MIT

# !/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
This script analysis LUCAS-15 dataset based on Fully connected neural network

written by : Mohammadmehdi Saberioon
revised date: 08.12.2020

"""
import random

import pandas as pd
# import numpy as np
import sys
import os
import argparse
import metrics
import tensorflow as tf


def parse_arg():
    parser = argparse.ArgumentParser(description='Fully connected Neural Networks')

    return parser.parse_args()


def Seperating_data_set(data_frame: pd.DataFrame) -> (pd.DataFrame, pd.DataFrame):
    """
    Prepare the raw data by :
        - Seperating dataset to two cluster namely calibration and testing


    Returns:
        Two panda DataFrames for Calibration and testing
    """
    data_frame_cal = data_frame.set_index("split").loc["calibration"]
    data_frame_tst = data_frame.set_index("split").loc["test"]
    return data_frame_cal, data_frame_tst


def splitting_dataset(data_frame: pd.DataFrame) -> (pd.DataFrame, pd.DataFrame):
    """
    Split data for dependant and independent variables : X_train and Y_train
    :param data_frame

    :return
        two pandas dataFrame: df_x and df_y

    """
    data_frame_x = data_frame.iloc[:, 6:]
    data_frame_y = data_frame.iloc[:, 5]

    return data_frame_x, data_frame_y


def build_fnn(data_frame_x: pd.DataFrame):
    """
    Building fully connected neural network (FNN) on data

    Returns:
        model

    """
    fnn_model = tf.keras.models.Sequential()
    fnn_model.add(tf.keras.layers.Dense(units=data_frame_x.shape[1],
                                        activation='relu',
                                        input_shape=[data_frame_x.shape[1]]))
    fnn_model.add(tf.keras.layers.Dropout(0.2))
    fnn_model.add(tf.keras.layers.Dense(1))

    # fnn_model.summary()

    fnn_model.compile(optimizer='adam', loss='mse')

    # early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=25)

    return fnn_model


def perform_analysis():
    """



    :return:
    """

    if len(sys.argv) == 1:  # no arguments, so print help message
        print("""Usage: python script.py data_path program_input out_path""")
        return

    dir_in = os.getcwd()
    dir_out = os.getcwd()

    try:
        dir_in = sys.argv[1]
        dir_out = sys.argv[2]
    except:
        print("Parameters: path/to/simple/file  input/folder  output/folder")
        sys.exit(0)

    df = pd.read_csv(dir_in)
    (cal_df, tst_df) = Seperating_data_set(df)

    (x_train, y_train) = splitting_dataset(cal_df)
    (x_test, y_test) = splitting_dataset(tst_df)

    print(x_train, y_train)
    print(x_test, y_test)

    model = build_fnn(x_train)

    model.summary()
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=25)
    random.seed(1)
    model.fit(x_train, y_train, batch_size=20, epochs=400, validation_data=(x_test, y_test), callbacks=[early_stop])

    fnn_losses = pd.DataFrame(model.history.history)

    predictions = model.predict(x_test)

    print("MSE:", metrics.MSE(y_test, predictions))
    print("RMSE:", metrics.RMSE(y_test, predictions))
    print("R-square:", metrics.R2(y_test, predictions))


# Main entry point
if __name__ == "__main__":
    perform_analysis()
