# SPDX-FileCopyrightText: `2020 Helmholtz Centre Potsdam - GFZ German Research Centre for Geosciences Potsdam, Germany
# SPDX-License-Identifier: MIT

import unittest
import pandas as pd
from src import merging_data


class Test_Prepare_data_set(unittest.TestCase):

    def test_return_type(self):
        birth_date = pd.Timestamp('1950-01-01')
        self.assertEqual(type(astronauts_analysis.calculate_age(birth_date)), int)

    def test_arg_type(self):
        birth_date = '1950-01-01'
        with self.assertRaises(AttributeError):
            astronauts_analysis.calculate_age(birth_date)


if __name__ == '__main__':
    unittest.main()
