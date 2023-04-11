import shutil

import ggsolver.ioutils as io
import unittest
import pathlib


class TestJSON(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Create folder if it does not exist.
        if not pathlib.Path("out/").exists():
            pathlib.Path("out/").mkdir()

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree("out/")

    def setUp(self):
        self.obj_dict = {"list": [1, 2, 3], "tuple": (4, 5, 6), "dict": {"t": "v"}}
        self.fpath = pathlib.Path("out/test.pkl").absolute().name

    def test_pickling(self):
        io.to_pickle(self.fpath, self.obj_dict)
        loaded_dict = io.from_pickle(self.fpath)
        self.assertEqual(self.obj_dict, loaded_dict)
