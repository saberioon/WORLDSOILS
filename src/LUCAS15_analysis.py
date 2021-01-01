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
from pathlib import Path
# import numpy as np
import sys
import os
import argparse
import metrics
import matplotlib.pyplot as plt
import tensorflow as tf

plt.style.use("ggplot")
_OUTPUT_PATH = "script-out"


def parse_arg():
    parser = argparse.ArgumentParser(prog='LUCAS15_analysis.py', description='Fully connected neural network for WORLDSOIL')
    parser.add_argument("-i", "--input", dest='input', type=str, help="", required=True)
    parser.add_argument("-o", "--output", dest='output', type=str, help="output filename ")
    parser.add_argument("-b", "--batchSize", dest="batchSize", type= int, help="batch size value", default=20)
    parser.add_argument("-e", "--epochs", dest="epochSize", type=int, help="epochs value", default=400)
    parser.add_argument("-d", "--dropout", dest="dropOut", type=int, help="Dropout value", default=0.2)
    parser.add_argument("-l", "--layer", dest="hiddenLayers", type=int, help="number of hidden layers", default=3)

    return parser


def separating_data_set(data_frame: pd.DataFrame) -> (pd.DataFrame, pd.DataFrame):
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
    data_frame_x = data_frame.iloc[:, 5:]
    data_frame_y = data_frame.iloc[:, 4]

    return data_frame_x, data_frame_y


def build_fnn(data_frame_x: pd.DataFrame):
    """
    Building fully connected neural network (FNN) on data

    Returns:
        model

    """
    parser = parse_arg()
    args = parser.parse_args()

    fnn_model = tf.keras.models.Sequential()
    # input layer + first hidden layer
    fnn_model.add(tf.keras.layers.Dense(units=data_frame_x.shape[1],
                                        activation='relu',
                                        input_shape=[data_frame_x.shape[1]]))
    fnn_model.add(tf.keras.layers.Dropout(args.dropOut))

    # second hidden layer
    fnn_model.add(tf.keras.layers.Dense(units=data_frame_x.shape[1], activation='relu'))
    fnn_model.add(tf.keras.layers.Dropout(args.dropOut))

    # third hidden layer
    fnn_model.add(tf.keras.layers.Dense(units=data_frame_x.shape[1], activation='relu'))
    fnn_model.add(tf.keras.layers.Dropout(args.dropOut))

    fnn_model.add(tf.keras.layers.Dense(1))

    # fnn_model.summary()

    fnn_model.compile(optimizer='adam', loss='mse')

    # early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=25)

    return fnn_model


def create_prediction_plot(obs_data_frame: pd.DataFrame, prd_data_frame: pd.DataFrame):
    """

    """
    fig, axs = plt.subplots(nrows=1,
                            ncols=1,
                            sharex=True)
    axs.scatter(obs_data_frame, prd_data_frame, color='blue')
    axs.plot(obs_data_frame, obs_data_frame, color='red')
    axs.set_title("title")
    axs.set_xlabel("X_axis")
    axs.set_ylabel("Y_axis")
    save(fig, "FNN_prediction.png")


def create_losses_plot(data_frame_losses: pd.DataFrame):
    """

    :return:
    """
    fig, axs = plt.subplots(nrows=1,
                            ncols=1,
                            sharex=True)
    axs.plot(data_frame_losses)
    axs.set_title("title")
    axs.set_xlabel("X_axis")
    axs.set_ylabel("Y_axis")
    save(fig, "FNN_losses.png")


def save(fig: plt.Figure, filename: str):
    """
    Saves a matplotlib Figure to a file. It overwrites existing files with the same filename.

    Args:
    fig: matplotlib.pyplot.Figure
    filename: str
    """
    fig.savefig(Path(_OUTPUT_PATH).resolve() / Path(filename))


def perform_analysis():
    """



    :return:
    """
    parser = parse_arg()
    args = parser.parse_args()

    # if len(sys.argv) == 1:  # no arguments, so print help message
    #     print("""Usage: python script.py data_path program_input out_path""")
    #     return
    #
    # dir_in = os.getcwd()
    # dir_out = os.getcwd()
    #
    # try:
    #     dir_in = sys.argv[1]
    #     dir_out = sys.argv[2]
    # except:
    #     print("Parameters: path/to/simple/file  input/folder  output/folder")
    #     sys.exit(0)

    #df = pd.read_csv(args.dir_in)
    df = pd.read_csv(args.input)
    (cal_df, tst_df) = separating_data_set(df)

    (x_train, y_train) = splitting_dataset(cal_df)
    (x_test, y_test) = splitting_dataset(tst_df)

    print(x_train)
    print(y_train)
    print(x_test)
    print(y_test)

    model = build_fnn(x_train)

    model.summary()
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=25)
    random.seed(1)
    model.fit(x_train, y_train, batch_size=args.batchSize, epochs=args.epochSize, validation_data=(x_test, y_test), callbacks=[early_stop])

    fnn_losses = pd.DataFrame(model.history.history)
    create_losses_plot(fnn_losses)

    predictions = model.predict(x_test)

    print("MSE:", metrics.MSE(y_test, predictions))
    print("RMSE:", metrics.RMSE(y_test, predictions))
    print("R-square:", metrics.R2(y_test, predictions))
    print("RPD:", metrics.RPD(y_test, predictions))

    create_prediction_plot(y_test, predictions)


# Main entry point
if __name__ == "__main__":
    perform_analysis()
