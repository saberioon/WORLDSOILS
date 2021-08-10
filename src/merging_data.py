# SPDX-FileCopyrightText:`2020 Helmholtz Centre Potsdam - GFZ German Research Centre for Geosciences Potsdam, Germany'
# SPDX-License-Identifier: EUPL-1.2

# !/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
This script merging and assiging split index (calibration, test) which aquired based on CLHS

written by : Mohammadmehdi Saberioon,PhD.
revised date: 11.12.2020

"""

import pandas as pd
import sys
import os
import argparse


def parse_arg():
    parser = argparse.ArgumentParser(prog='merging_data.py', description='Merging cLHS and resampled data')
    parser.add_argument("-i", "--input1", dest='cLHSref', type=str, help=" cLHS file", required=True)
    parser.add_argument("-b", "--input2", dest='directory', type=str, help=" input filename ", required=True)
    parser.add_argument("-o", "--output", dest='output', type=str, help="output filename ", required=True)

    return parser


def prepare_data_set(df1, df2: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare the raw data by :
        - merging data according to cLHS
        - splitting to calibration and test

    Arg:
        df1: A panda DataFrame/ spliting information.
        df2: A panda DataFram/ LUCAS data

    Returns:
        One panda DataFrames with column split :1.Calibration 2.Test
    """

    df1.index.name = df2.index.name = 'ID'
    data_frame = df1.merge(df2, left_index=True, right_index=True)

    return data_frame


def run():
    parser = parse_arg()
    args = parser.parse_args()


    # if len(sys.argv) == 1:  # no arguments, so print help message
    #     print("""Usage: python script.py data_path program_input out_path""")
    #     return
    # dir_in_f1 = os.getcwd()
    # dir_in_f2 = os.getcwd()
    # dir_out = os.getcwd()
    #
    # try:
    #     dir_in_f1 = sys.argv[1]
    #     dir_in_f2 = sys.argv[2]
    #     dir_out = sys.argv[3]
    # except:
    #     print("Parameters: path/to/simple/file  input/folder  output/folder")
    #     sys.exit(0)

    # df1 = pd.read_csv(dir_in_f1).set_index("ID")
    # df2 = pd.read_csv(dir_in_f2).set_index("PointID")

    df1 = pd.read_csv(args.cLHSref).set_index("ID")
    df2 = pd.read_csv(args.directory).set_index("PointID")

    df_out = prepare_data_set(df1, df2)
    # df_out.to_csv(dir_out, index=True)
    df_out.to_csv(args.output, index=True)


if __name__ == "__main__":
    run()
    print("DONE!! Be Happy ;)")
