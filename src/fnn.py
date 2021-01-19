# SPDX-FileCopyrightText:`2020 Helmholtz Centre Potsdam - GFZ German Research Centre for Geosciences Potsdam, Germany'
# SPDX-License-Identifier: MIT

# !/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
This script analysis LUCAS-15/bssl dataset based on Fully connected neural network

written by : Mohammadmehdi Saberioon
revised date: 01.01.2021

"""
import random

import pandas as pd
from pathlib import Path
import numpy as np
import sys
import os
import argparse
import metrics
import matplotlib.pyplot as plt
# import tensorflow as tf
from version import __version__
import preProcessing

plt.style.use("ggplot")
_OUTPUT_PATH = "script-out"


def parse_arg():
    parser = argparse.ArgumentParser(prog='fnn.py',
                                     description='Fully connected neural network for WORLDSOIL')
    parser.add_argument("-i", "--input", dest='input', type=str, help="", required=True)
    parser.add_argument("-o", "--output", dest='output', type=str, help="output filename ")
    parser.add_argument("-b", "--batchSize", dest="batchSize", type=int, help="batch size value", default=20)
    parser.add_argument("-e", "--epochs", dest="epochSize", type=int, help="epochs value", default=400)
    parser.add_argument("-d", "--dropout", dest="dropOut", type=float, help="Dropout value", default=0.2)
    parser.add_argument("-l", "--layer", dest="hiddenLayers", type=int, help="number of hidden layers", default=3)
    parser.add_argument("-v", "--version", action="version", version="%(prog)s " + __version__)
    parser.add_argument("-r", "--learnRate", dest="learnRate", type=float, help="set the learning rate", default=0.01)
    parser.add_argument("-k", "--kernel", dest="kernel", type=str, help="set kernel initializer", default="glorot_uniform")

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
    # # for full dataset
    # data_frame_x = data_frame.iloc[:, 5:]
    # # data_frame_y = data_frame.iloc[:, 4]
    # data_frame_y = data_frame['OC'].values

    # For dataset witout outliers (clean )
    data_frame_x = data_frame.iloc[:, 6:]
    # data_frame_y = data_frame.iloc[:, 4]
    data_frame_y = data_frame['OC'].values

    return data_frame_x, data_frame_y


def build_fnn_5l(data_frame_x: pd.DataFrame):
    """
    Building fully connected neural network (FNN) on data with five hidden layers

    Returns:
        model

    """
    parser = parse_arg()
    args = parser.parse_args()

    fnn_model_5l = tf.keras.models.Sequential()
    # input layer + first hidden layer
    fnn_model_5l.add(tf.keras.layers.Dense(units=data_frame_x.shape[1], input_shape=[data_frame_x.shape[1]],
                                           kernel_initializer=args.kernel))
    fnn_model_5l.add(tf.keras.layers.BatchNormalization())
    fnn_model_5l.add(tf.keras.layers.LeakyReLU(alpha=0.01))
    fnn_model_5l.add(tf.keras.layers.Dropout(args.dropOut))

    # second hidden layer
    fnn_model_5l.add(tf.keras.layers.Dense(units=data_frame_x.shape[1], kernel_initializer=args.kernel))
    fnn_model_5l.add(tf.keras.layers.BatchNormalization())
    fnn_model_5l.add(tf.keras.layers.LeakyReLU(alpha=0.01))
    fnn_model_5l.add(tf.keras.layers.Dropout(args.dropOut))

    # third hidden layer
    fnn_model_5l.add(tf.keras.layers.Dense(units=data_frame_x.shape[1], kernel_initializer=args.kernel))
    fnn_model_5l.add(tf.keras.layers.BatchNormalization())
    fnn_model_5l.add(tf.keras.layers.LeakyReLU(alpha=0.01))
    fnn_model_5l.add(tf.keras.layers.Dropout(args.dropOut))

    # fourth hidden layer
    fnn_model_5l.add(tf.keras.layers.Dense(units=data_frame_x.shape[1], kernel_initializer=args.kernel))
    fnn_model_5l.add(tf.keras.layers.BatchNormalization())
    fnn_model_5l.add(tf.keras.layers.LeakyReLU(alpha=0.01))
    fnn_model_5l.add(tf.keras.layers.Dropout(args.dropOut))

    # fifth hidden layer
    fnn_model_5l.add(tf.keras.layers.Dense(units=data_frame_x.shape[1], kernel_initializer=args.kernel))
    fnn_model_5l.add(tf.keras.layers.BatchNormalization())
    fnn_model_5l.add(tf.keras.layers.LeakyReLU(alpha=0.01))
    fnn_model_5l.add(tf.keras.layers.Dropout(args.dropOut))

    fnn_model_5l.add(tf.keras.layers.Dense(1))

    # fnn_model.summary()

    # opt = tf.keras.optimizers.SGD(learning_rate=0.0001, nesterov=True)
    opt = tf.keras.optimizers.Adam(learning_rate=args.learnRate)
    fnn_model_5l.compile(optimizer=opt, loss='mse', metrics=[tf.keras.metrics.RootMeanSquaredError()])

    # early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=25)

    return fnn_model_5l


def build_fnn_4l(data_frame_x: pd.DataFrame):
    """
    Building fully connected neural network (FNN) on data with four hidden layers

    Returns:
        model

    """
    parser = parse_arg()
    args = parser.parse_args()

    fnn_model_4l = tf.keras.models.Sequential()
    # input layer + first hidden layer
    fnn_model_4l.add(tf.keras.layers.Dense(units=data_frame_x.shape[1], input_shape=[data_frame_x.shape[1]],
                                           kernel_initializer=args.kernel))
    fnn_model_4l.add(tf.keras.layers.BatchNormalization())
    fnn_model_4l.add(tf.keras.layers.LeakyReLU(alpha=0.1))
    fnn_model_4l.add(tf.keras.layers.Dropout(args.dropOut))

    # second hidden layer
    fnn_model_4l.add(tf.keras.layers.Dense(units=data_frame_x.shape[1], kernel_initializer=args.kernel))
    fnn_model_4l.add(tf.keras.layers.BatchNormalization())
    fnn_model_4l.add(tf.keras.layers.LeakyReLU(alpha=0.01))
    fnn_model_4l.add(tf.keras.layers.Dropout(args.dropOut))

    # third hidden layer
    fnn_model_4l.add(tf.keras.layers.Dense(units=data_frame_x.shape[1], kernel_initializer=args.kernel))
    fnn_model_4l.add(tf.keras.layers.BatchNormalization())
    fnn_model_4l.add(tf.keras.layers.LeakyReLU(alpha=0.01))
    fnn_model_4l.add(tf.keras.layers.Dropout(args.dropOut))

    # fourth hidden layer
    fnn_model_4l.add(tf.keras.layers.Dense(units=data_frame_x.shape[1], kernel_initializer=args.kernel))
    fnn_model_4l.add(tf.keras.layers.BatchNormalization())
    fnn_model_4l.add(tf.keras.layers.LeakyReLU(alpha=0.01))
    fnn_model_4l.add(tf.keras.layers.Dropout(args.dropOut))

    fnn_model_4l.add(tf.keras.layers.Dense(1))

    # fnn_model.summary()

    # opt = tf.keras.optimizers.SGD(learning_rate=0.0001, nesterov=True)
    opt = tf.keras.optimizers.Adam(learning_rate=args.learnRate)
    fnn_model_4l.compile(optimizer=opt, loss='mse', metrics=[tf.keras.metrics.RootMeanSquaredError()])

    # early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=25)

    return fnn_model_4l


def build_fnn_3l(data_frame_x: pd.DataFrame):
    """
    Building fully connected neural network (FNN) on data with three hidden layers

    Returns:
        model

    """
    parser = parse_arg()
    args = parser.parse_args()

    fnn_model = tf.keras.models.Sequential()
    # input layer + first hidden layer
    fnn_model.add(tf.keras.layers.Dense(units=data_frame_x.shape[1],
                                        input_shape=[data_frame_x.shape[1]],
                                        kernel_initializer=args.kernel))
    fnn_model.add(tf.keras.layers.BatchNormalization())
    fnn_model.add(tf.keras.layers.LeakyReLU(alpha=0.01))
    fnn_model.add(tf.keras.layers.Dropout(args.dropOut))

    # second hidden layer
    fnn_model.add(tf.keras.layers.Dense(units=data_frame_x.shape[1],
                                        kernel_initializer=args.kernel))
    fnn_model.add(tf.keras.layers.BatchNormalization())
    fnn_model.add(tf.keras.layers.LeakyReLU(alpha=0.01))
    fnn_model.add(tf.keras.layers.Dropout(args.dropOut))

    # third hidden layer
    fnn_model.add(tf.keras.layers.Dense(units=data_frame_x.shape[1],
                                        kernel_initializer=args.kernel))
    fnn_model.add(tf.keras.layers.BatchNormalization())
    fnn_model.add(tf.keras.layers.LeakyReLU(alpha=0.01))
    fnn_model.add(tf.keras.layers.Dropout(args.dropOut))

    fnn_model.add(tf.keras.layers.Dense(1))

    # fnn_model.summary()
    # opt = tf.keras.optimizers.SGD(learning_rate=0.0001, nesterov=True)
    opt = tf.keras.optimizers.Adam(learning_rate=args.learnRate)
    fnn_model.compile(optimizer=opt, loss='mse', metrics=[tf.keras.metrics.RootMeanSquaredError()])

    # early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=25)

    return fnn_model


def build_fnn_2l(data_frame_x: pd.DataFrame):
    """
    Building fully connected neural network (FNN) on data with two hidden layers

    Returns:
        model

    """
    parser = parse_arg()
    args = parser.parse_args()

    fnn_model_2l = tf.keras.models.Sequential()
    # input layer + first hidden layer
    fnn_model_2l.add(tf.keras.layers.Dense(units=data_frame_x.shape[1],
                                           input_shape=[data_frame_x.shape[1]],
                                           kernel_initializer=args.kernel))
    fnn_model_2l.add(tf.keras.layers.LeakyReLU(alpha=0.01))
    fnn_model_2l.add(tf.keras.layers.Dropout(args.dropOut))

    # second hidden layer
    fnn_model_2l.add(tf.keras.layers.Dense(units=data_frame_x.shape[1],
                                           kernel_initializer=args.kernel))
    fnn_model_2l.add(tf.keras.layers.LeakyReLU(alpha=0.01))
    fnn_model_2l.add(tf.keras.layers.Dropout(args.dropOut))

    fnn_model_2l.add(tf.keras.layers.Dense(1))

    # fnn_model.summary()
    opt = tf.keras.optimizers.SGD(learning_rate=args.learnRate, nesterov=True)
    # opt = tf.keras.optimizers.Adam(learning_rate=0.001)
    fnn_model_2l.compile(optimizer=opt, loss='mse', metrics=[tf.keras.metrics.RootMeanSquaredError()])

    # early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=25)

    return fnn_model_2l


def build_fnn_1l(data_frame_x: pd.DataFrame):
    """
    Building fully connected neural network (FNN) on data with one hidden layers

    Returns:
        model

    """
    parser = parse_arg()
    args = parser.parse_args()

    fnn_model_1l = tf.keras.models.Sequential()
    # input layer + first hidden layer
    fnn_model_1l.add(tf.keras.layers.Dense(units=data_frame_x.shape[1],
                                           input_shape=[data_frame_x.shape[1]],
                                           kernel_initializer=args.kernel))
    fnn_model_1l.add(tf.keras.layers.LeakyReLU(alpha=0.01))
    fnn_model_1l.add(tf.keras.layers.Dropout(args.dropOut))

    # fnn_model.summary()

    # output layer
    fnn_model_1l.add(tf.keras.layers.Dense(1))

    # compile layers
    opt = tf.keras.optimizers.SGD(learning_rate=args.learnRate, nesterov=True)
    # opt = tf.keras.optimizers.Adam(learning_rate=0.001)
    fnn_model_1l.compile(optimizer=opt, loss='mse', metrics=[tf.keras.metrics.RootMeanSquaredError()])

    # early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=25)

    return fnn_model_1l


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


def trained_model_save(model, filename: str):
    """

    :return:
    """
    tf.keras.models.save_model(model, filepath=Path(_OUTPUT_PATH).resolve() / Path(filename), overwrite=True)


def perform_analysis():
    """



    :return:
    """
    parser = parse_arg()
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    # df = df.apply(lambda x: preProcessing.scaling_y_data(x) if x.name == 'OC' else x)  # scaling OC data

    (cal_df, tst_df) = separating_data_set(df)

    (X_train, y_train) = splitting_dataset(cal_df)
    (X_test, y_test) = splitting_dataset(tst_df)

    # Scale the features
    X_train = preProcessing.scaler_min_max_x_data(X_train)
    X_test = preProcessing.scaler_min_max_x_data(X_test)

    y_train = preProcessing.scaler_min_max_y_data(y_train)
    y_test = preProcessing.scaler_min_max_y_data(y_test)

    print(X_train)
    print(y_train)
    print(X_test)
    print(y_test)

    print(X_train.shape)
    print(y_train.shape)

    if args.hiddenLayers == 5:
        model = build_fnn_5l(X_train)
    elif args.hiddenLayers == 4:
        model = build_fnn_4l(X_train)
    elif args.hiddenLayers == 3:
        model = build_fnn_3l(X_train)
    elif args.hiddenLayers == 2:
        model = build_fnn_2l(X_train)
    else:
        model = build_fnn_1l(X_train)

    model.summary()

    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=25)
    random.seed(1)
    model.fit(X_train, y_train, batch_size=args.batchSize, epochs=args.epochSize, validation_data=(X_test, y_test),
              callbacks=[early_stop])

    fnn_losses = pd.DataFrame(model.history.history)
    create_losses_plot(fnn_losses)

    trained_model_save(model, "trained_model.h5")

    predictions = model.predict(X_test)

    print("MSE:", metrics.MSE(y_test, predictions))
    print("RMSE:", metrics.RMSE(y_test, predictions))
    print("R-square:", metrics.R2(y_test, predictions))
    print("RPD:", metrics.RPD(y_test, predictions))

    create_prediction_plot(y_test, predictions)


# Main entry point
if __name__ == "__main__":
    perform_analysis()
