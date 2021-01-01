import unittest
from version import __version__
import os


class Testfnn(unittest.TestCase):

    def test_main(self):
        print(__version__)


if __name__ == "__main__":
    unittest.main()
