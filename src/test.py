# SPDX-FileCopyrightText:`2020 Helmholtz Centre Potsdam - GFZ German Research Centre for Geosciences Potsdam, Germany'
# SPDX-License-Identifier: EUPL-1.2


"""
This script analysis LUCAS-15/bssl dataset based on Fully connected neural network

written by : Mohammadmehdi Saberioon
revised date: 01.01.2021

"""

# !/usr/bin/env python3
# -*- coding: utf-8 -*-

import unittest
from version import __version__
import os
import pandas.util.testing as pdt


class Testfnn(unittest.TestCase):

    def test_main(self):
        print(__version__)



if __name__ == "__main__":
    unittest.main()
